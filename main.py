import json
import os.path
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from mpl_toolkits.basemap import Basemap

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
    df['POLYLINE'] = df['POLYLINE'].map(lambda x: json.loads(x))
    df['POLYLINE'] = df['POLYLINE'].map(lambda x: np.array(x))
    df['KNOWN_TRIP_LENGTH'] = df['POLYLINE'].map(lambda x: len(x)) * 15
    df.drop(df[df['KNOWN_TRIP_LENGTH'] < 30].index, inplace=True)
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
        df['KNOWN_TRIP_LENGTH'] = df['KNOWN_TRIP_LENGTH'].map(lambda x: x - 15)
        df.drop(['DEST'], axis=1, inplace=True)
        for column in ['ORIGIN_LON', 'ORIGIN_LAT', 'DEST_LON', 'DEST_LAT']:
            df = remove_outliers(df, column)
    
    df['KNOWN_DISTANCE'] = df['POLYLINE'].map(lambda x: haversine(x[0], x[-1]))
    df['KNOWN_BEARING'] = df['POLYLINE'].map(lambda x: bearing(x[0], x[-1]))
    
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

def main():
    if os.path.isfile('./data/train.p'):
        print('Reading Pickle')
        df = pd.read_pickle('./data/train.p')
    else:
        print('Reading CSV')
        df = pd.read_csv('./data/train.csv')
        print('Procesing DF')
        df = process_df(df)
        print('Saving to Pickle')
        df.to_pickle('./data/train.p')

    print('Mapping DF')
    explore(df)
    
    return df

if __name__ == '__main__':
    df = main()