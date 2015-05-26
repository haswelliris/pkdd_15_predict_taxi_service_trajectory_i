import json
import os.path
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from mpl_toolkits.basemap import Basemap
from sklearn import cross_validation, preprocessing
from sklearn.ensemble import ExtraTreesRegressor

def haversine(start, end):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1 = start
    lon2, lat2 = end
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def bearing(start, end):
    lon1, lat1 = start
    lon2, lat2 = end

    lon1 = radians(lon1)
    lat1 = radians(lat1)
    lon2 = radians(lon2)
    lat2 = radians(lat2)

    long_diff = lon2 - lon1
    x = sin(long_diff) * cos(lat2)
    y = cos(lat1) * sin(lat2) - (sin(lat1) * cos(lat2) * cos(long_diff))

    initial_bearing = atan2(x, y)

    initial_bearing = degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    
    return compass_bearing

def process_df(df, train=True):
    df.drop(df[df['MISSING_DATA'] == True].index, inplace=True)
    call_map = {'A': 1, 'B': 2, 'C': 3}
    df['CALL_TYPE'] = df['CALL_TYPE'].map(call_map)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
    df['DATE'] = df['TIMESTAMP'].dt.date.map(dt.date.toordinal)
    df['YEAR'] = df['TIMESTAMP'].dt.year
    df['MONTH'] = df['TIMESTAMP'].dt.month
    df['WEEK'] = df['TIMESTAMP'].dt.week
    df['DAY'] = df['TIMESTAMP'].dt.day
    df['HOUR'] = df['TIMESTAMP'].dt.hour
    df['MINUTE'] = df['TIMESTAMP'].dt.minute
    df['MINUTEOFDAY'] = df['HOUR'] * 60 + df['MINUTE']
    df['SECOND'] = df['TIMESTAMP'].dt.second
    df['SECONDOFDAY'] = df['MINUTEOFDAY'] * 60 + df['SECOND']
    df['WEEKDAY'] = df['TIMESTAMP'].dt.weekday
    df['POLYLINE'] = df['POLYLINE'].map(json.loads)
    df['POLYLINE'] = df['POLYLINE'].map(np.array)
    df['KNOWN_DURATION'] = df['POLYLINE'].map(len) * 15
    
    if train:
        df.drop(df[df['KNOWN_DURATION'] < 30].index, inplace=True)
    
    df['ORIGIN'] = df['POLYLINE'].map(lambda x: x[0])
    df['ORIGIN_LON'] = df['ORIGIN'].map(lambda x: x[0])
    df['ORIGIN_LAT'] = df['ORIGIN'].map(lambda x: x[1])
    df['ORIGIN_CALL'].fillna(0, inplace=True)
    df['ORIGIN_STAND'].fillna(0, inplace=True)

    if train:
        df['DEST'] = df['POLYLINE'].map(lambda x: x[-1])
        df['DEST_LON'] = df['DEST'].map(lambda x: x[0])
        df['DEST_LAT'] = df['DEST'].map(lambda x: x[1])
        df['POLYLINE'] = df['POLYLINE'].map(lambda x: x[:-1])
        df['KNOWN_DURATION'] = df['KNOWN_DURATION'] - 15
        df.drop(['DEST', 'TRIP_ID'], axis=1, inplace=True)
        for column in ['ORIGIN_LON', 'ORIGIN_LAT', 'DEST_LON', 'DEST_LAT']:
            df = remove_outliers(df, column)
    
    df['KNOWN_DISTANCE'] = df['POLYLINE'].map(lambda x: haversine(x[0], x[-1]))
    df['KNOWN_BEARING'] = df['POLYLINE'].map(lambda x: bearing(x[0], x[-1]))

    try:
        print('Calculated average distance for short trips: {}'.format(process_df.average_short_distance))
        print('Calculated average bearing for short trips in the left half: {}'.format(process_df.average_short_bearing_left))
        print('Calculated average bearing for short trips in the right half: {}'.format(process_df.average_short_bearing_right))
    except AttributeError:
        process_df.mid_lon = df['ORIGIN_LON'].mean()
        process_df.average_short_distance = df.loc[df['KNOWN_DURATION'] == 30, 'KNOWN_DISTANCE'].mean() / 2
        process_df.average_short_bearing_left = df.loc[df['ORIGIN_LON'] < process_df.mid_lon, 'KNOWN_BEARING'].mean()
        process_df.average_short_bearing_right = df.loc[df['ORIGIN_LON'] >= process_df.mid_lon, 'KNOWN_BEARING'].mean()
    finally:
        df.loc[df['KNOWN_DURATION'] == 15, 'KNOWN_DISTANCE'] = process_df.average_short_distance
        df.loc[(df['KNOWN_DURATION'] == 15) & (df['ORIGIN_LON'] < process_df.mid_lon), 'KNOWN_BEARING'] = process_df.average_short_bearing_left
        df.loc[(df['KNOWN_DURATION'] == 15) & (df['ORIGIN_LON'] >= process_df.mid_lon), 'KNOWN_BEARING'] = process_df.average_short_bearing_right



    df.drop(['DAY_TYPE', 'MISSING_DATA', 'ORIGIN', 'POLYLINE', 'TIMESTAMP'], axis=1, inplace=True)

    return df

def remove_outliers(df, column):
    summary = df[column].describe(percentiles=[0.01, 0.99])
    return df[(df[column] >= summary['1%']) & (df[column] <= summary['99%'])]

def draw_map(lon_column, lat_column, subplot=1, cmap='Blues'):
    plt.subplot(2, 1, subplot)
    m = Basemap(projection='merc', resolution = 'h', urcrnrlon=-8.5363740000000004, urcrnrlat=41.237622000000002, llcrnrlon=-8.6920289999999998, llcrnrlat=41.112071999999998)
    m.readshapefile('./data/roads', 'landmarks')
    m.drawcoastlines()
    m.drawrivers()
    m.drawcountries()
    m.drawmapboundary()

    lon_bins = np.linspace(lon_column.min() - 1, lon_column.max() + 1, 500)
    lat_bins = np.linspace(lat_column.min() - 1, lat_column.max() + 1, 500)

    density, _, _ = np.histogram2d(lat_column, lon_column, [lat_bins, lon_bins])

    lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)

    xs, ys = m(lon_bins_2d, lat_bins_2d)

    plt.pcolormesh(xs, ys, density, cmap=plt.get_cmap(cmap))
    plt.colorbar(orientation='horizontal')

    m.scatter(lon_column.tolist(), lat_column.tolist(), alpha=0.00125, latlon=True, c='green')

    m.drawmapscale(-8.685, 41.14, -8.685, 41.14, 1)

def explore(df):
    draw_map(df['ORIGIN_LON'], df['ORIGIN_LAT'])
    draw_map(df['DEST_LON'], df['DEST_LAT'], 2, 'Reds')

def main(exp=False):
    if os.path.isfile('./data/data.hdf'):
        print('Reading HDF')
        train_df = pd.read_hdf('./data/data.hdf', 'train')
        test_df = pd.read_hdf('./data/data.hdf', 'test')
    else:
        print('Reading Train CSV')
        train_df = pd.read_csv('./data/train.csv')
        print('Procesing Train DF')
        train_df = process_df(train_df)
        print('Saving Train to HDF')
        train_df.to_hdf('./data/data.hdf', 'train')

        print('Reading Test CSV')
        test_df = pd.read_csv('./data/test.csv')
        print('Procesing Test DF')
        test_df = process_df(test_df, False)
        assert len(test_df) == 320
        print('Saving Test to HDF')
        test_df.to_hdf('./data/data.hdf', 'test')

    if exp:
        print('Mapping Train DF')
        explore(train_df)

    print('Training and running model')
    scaler = preprocessing.StandardScaler()
    train_data = train_df.drop(['DEST_LON', 'DEST_LAT'], axis=1).values
    train_data = scaler.fit_transform(train_data)
    target_data = np.array([train_df['DEST_LON'], train_df['DEST_LAT']]).T

    train_data, cv_data, target_data, cv_target_data = cross_validation.train_test_split(
        train_data, target_data, test_size=0.2)

    clf = ExtraTreesRegressor(n_estimators=100, n_jobs=-1)
    clf.fit(train_data, target_data)
    cv_predictions = clf.predict(cv_data)

    results = []
    for start, end in zip(cv_predictions, cv_target_data):
        results.append(haversine(start, end))

    print('Average estimate was {} km off.'.format(sum(results)/len(results)))

    test_data = scaler.transform(test_df.drop(['TRIP_ID'], axis=1).values)

    predictions = clf.predict(test_data)

    with open('./data/output.csv', 'w') as o:
        o.write('TRIP_ID,LATITUDE,LONGITUDE\n')
        for trip_id, prediction in zip(test_df['TRIP_ID'], predictions):
            o.write('{},{},{}\n'.format(trip_id, prediction[1], prediction[0]))

    return locals()
        

if __name__ == '__main__':
    results = main()
    




