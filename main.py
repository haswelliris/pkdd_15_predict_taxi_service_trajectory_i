import ast
import pandas as pd
import numpy as np


def create_new_rows_from_polyline(df):
	new_df = pd.DataFrame(columns=df.columns)
	new_df['LONGITUDE'] = []
	new_df['LATITUDE'] = []
	new_df.drop('POLYLINE', axis=1, inplace=True)
	for i, row in df.iterrows():
		new_row = row.drop('POLYLINE')
		new_row_timestamp = new_row['TIMESTAMP']
		for loc in row['POLYLINE']:
			new_row['TIMESTAMP'] = new_row_timestamp
			new_row['LONGITUDE'] = loc[0]
			new_row['LATITUDE'] = loc[1]
			new_df = new_df.append(new_row)
			new_row_timestamp += 15

	return new_df

chunks = pd.read_csv('./data/train.csv', chunksize=1000)
store = pd.HDFStore('./data/data.hdf')

for chunk in chunks:
	chunk['POLYLINE'] = chunk['POLYLINE'].map(lambda x: ast.literal_eval(x))
	df = create_new_rows_from_polyline(chunk).convert_objects()
	print(df.head())
	store.append('train', df)
	store.flush()

store.close()