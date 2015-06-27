import json
import os.path
import datetime as dt
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EARTH_RADIUS = 6371

def haversine(start_lon, start_lat, end_lon, end_lat):
    start_lon = np.radians(start_lon)
    start_lat = np.radians(start_lat)
    end_lon = np.radians(end_lon)
    end_lat = np.radians(end_lat)

    dlon = end_lon - start_lon
    dlat = end_lat - start_lat

    a = np.sin(dlat/2)**2 + np.cos(start_lat) * np.cos(end_lat) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return np.nan_to_num(c * EARTH_RADIUS)

def bearing(start_lon, start_lat, end_lon, end_lat):
    start_lon = np.radians(start_lon)
    start_lat = np.radians(start_lat)
    end_lon = np.radians(end_lon)
    end_lat = np.radians(end_lat)

    dlon = end_lon - start_lon
    x = np.sin(dlon) * np.cos(end_lat)
    y = np.cos(start_lat) * np.sin(end_lat) - (np.sin(start_lat) * np.cos(end_lat) * np.cos(dlon))

    initial_bearing = np.arctan2(x, y)

    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return np.nan_to_num(compass_bearing)

def process_chunk(df):
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

    return df

def process_df(df, train=True):
    df_chunks = np.array_split(df, 4)
    with mp.Pool() as p:
        processed_chunks = p.map(process_chunk, df_chunks)
        df = pd.DataFrame([], columns=df.columns)
        for chunk in processed_chunks:
            df = df.append(chunk, ignore_index=True)

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
        df.drop(['DEST'], axis=1, inplace=True)
        for column in ['ORIGIN_LON', 'ORIGIN_LAT', 'DEST_LON', 'DEST_LAT']:
            df = remove_outliers(df, column)

    df['LAST_LOC'] = df['POLYLINE'].map(lambda x: x[-1])
    df['LAST_LON'] = df['LAST_LOC'].map(lambda x: x[0])
    df['LAST_LAT'] = df['LAST_LOC'].map(lambda x: x[1])

    df['KNOWN_DISTANCE'] = haversine(df['ORIGIN_LON'].values, df['ORIGIN_LAT'].values, df['LAST_LON'].values, df['LAST_LAT'].values)
    df['KNOWN_BEARING'] = bearing(df['ORIGIN_LON'].values, df['ORIGIN_LAT'].values, df['LAST_LON'].values, df['LAST_LAT'].values)

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

    df = add_closest_station(df)

    df.drop(['DAY_TYPE', 'MISSING_DATA', 'ORIGIN', 'LAST_LOC', 'POLYLINE', 'TIMESTAMP'], axis=1, inplace=True)

    return df

def remove_outliers(df, column):
    summary = df[column].describe(percentiles=[0.01, 0.99])
    return df[(df[column] >= summary['1%']) & (df[column] <= summary['99%'])]

def add_closest_station(df):
    if os.path.isfile('./data/data.hdf'):
        print('Reading Lookup HDF')
        lookup_df = pd.read_hdf('./data/data.hdf', 'lookup')
    else:
        print('Reading Lookup CSV')
        lookup_df = pd.read_csv('./data/metaData_taxistandsID_name_GPSlocation.csv')
        lookup_df = lookup_df.reindex(lookup_df['ID']).drop(['ID', 'Descricao'], axis=1).dropna()
        lookup_df.to_hdf('./data/data.hdf', 'lookup')

    print('Creating lookup')
    merged_df = pd.DataFrame(np.array(pd.tools.util.cartesian_product([df.index, lookup_df.index])).T, columns=['DF_INDEX', 'LOOKUP_INDEX'])
    merged_df = merged_df.merge(df[['ORIGIN_LON', 'ORIGIN_LAT']], left_on='DF_INDEX', right_index=True).merge(lookup_df, left_on='LOOKUP_INDEX', right_index=True)
    merged_df['DISTANCE'] = haversine(merged_df['ORIGIN_LON'], merged_df['ORIGIN_LAT'], merged_df['Longitude'], merged_df['Latitude'])

    print('Adding closest stations')
    closest_df = merged_df[merged_df.groupby('DF_INDEX')['DISTANCE'].transform(np.min) == merged_df['DISTANCE']].drop_duplicates(['DF_INDEX', 'DISTANCE'])
    closest_df.index = closest_df['DF_INDEX']
    df['CLOSEST_ID']  = closest_df['LOOKUP_INDEX']
    df['CLOSEST_DISTANCE'] = closest_df['DISTANCE']
    df['CLOSEST_LON'] = closest_df['Longitude']
    df['CLOSEST_LAT'] = closest_df['Latitude']

    return df

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

def heatmap():
    if os.path.isfile('./data/heatmap.png'):
        plt.imshow(plt.imread('./data/heatmap.png'))
    else:
        # https://www.kaggle.com/mcwitt/pkdd-15-predict-taxi-service-trajectory-i/heatmap
        polyline = pd.read_csv('./data/train.csv',
                               usecols=['POLYLINE'],
                               converters={'POLYLINE': lambda x: json.loads(x)})
        bins = 1000
        lat_min, lat_max = 41.04961, 41.24961
        lon_min, lon_max = -8.71099, -8.51099
        z = np.zeros((bins, bins))
        latlon = np.array([(lat, lon)
                           for path in polyline['POLYLINE']
                           for lon, lat in path if len(path) > 0])

        z += np.histogram2d(*latlon.T, bins=bins,
                            range=[[lat_min, lat_max],
                                   [lon_min, lon_max]])[0]

        log_density = np.log(1+z)

        plt.imshow(log_density[::-1,:], # flip vertically
                   extent=[lat_min, lat_max, lon_min, lon_max])

        plt.savefig('./data/heatmap.png')


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

    print('Preparing pipeline')
    scaler = StandardScaler()
    etr = ExtraTreesRegressor(n_estimators=100, n_jobs=-1)
    pipe = Pipeline([('scaler', scaler), ('etr', etr)])

    print('Training model')
    train_data = train_df.drop(['TRIP_ID', 'DEST_LON', 'DEST_LAT'], axis=1).values
    target_data = np.array([train_df['DEST_LON'], train_df['DEST_LAT']]).T
    test_data = test_df.drop(['TRIP_ID'], axis=1).values

    train_data, cv_data, target_data, cv_target_data = train_test_split(
        train_data, target_data, test_size=0.2)

    pipe.fit(train_data, target_data)

    print('Predicting CV results')
    cv_predictions = pipe.predict(cv_data)

    results = haversine(cv_target_data[:,0], cv_target_data[:,1], cv_predictions[:,0], cv_predictions[:,1])

    print('Average estimate was {} km off.'.format(results.mean()))

    print('Predicting test results')
    predictions = pipe.predict(test_data)

    with open('./data/output.csv', 'w') as o:
        o.write('TRIP_ID,LATITUDE,LONGITUDE\n')
        for trip_id, prediction in zip(test_df['TRIP_ID'], predictions):
            o.write('{},{},{}\n'.format(trip_id, prediction[1], prediction[0]))

    cv_df = pd.DataFrame(scaler.inverse_transform(cv_data),
        columns=train_df.drop(['TRIP_ID', 'DEST_LAT', 'DEST_LON'], axis=1).columns)

    cv_df['DEST_LON'] = cv_target_data[:,0]
    cv_df['DEST_LAT'] = cv_target_data[:,1]
    cv_df['PDEST_LON'] = cv_predictions[:,0]
    cv_df['PDEST_LAT'] = cv_predictions[:,1]
    cv_df['DELTA'] = haversine(cv_df['DEST_LON'], cv_df['DEST_LAT'],
        cv_df['PDEST_LON'], cv_df['PDEST_LAT'])

    return locals()


if __name__ == '__main__':
    results = main()





