import operator
import collections
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt

# Haversine formula for distance between two points on the earth (specified in decimal degrees)
def haversine(lon1,lat1,lon2,lat2):
	lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
	dlon = lon2 - lon1 
	dlat = lat2 - lat1 
	a = (sin(dlat/2)**2) + cos(lat1) * cos(lat2) * (sin(dlon/2)**2)
	c = 2 * asin(sqrt(a))
	r = 6371 
	
	return c*r # distance in km

# Given floor name get it's floor id
'''
Assumption that: (Used in ping info) --> (Used in store mapping)
				 Floor 1 --> Level 1, Floor 2 --> Level 2
				 Floor 3 --> Level 3, Floor 4 --> Level 4
'''
def getFloor_id(floor):
	if(floor == "Concourse"):
		return 0
	else:
		return int(floor.split(" ")[1])

# Get a list return elemnt with max freqency 
def getMax_freqency(list):
	freq = collections.Counter(list)
	return max(freq, key=freq.get)

def getShop_category(name):
	return category_mapping_matrix[np.where(category_mapping_matrix[:,0]==name)[0][0]][2]

# Loading the data into dataframes
demographics_data = pd.read_excel('../data/Data.xlsx',sheet_name='Demographics',index_col=0)
store_mapping_data = pd.read_excel('../data/Data.xlsx',sheet_name='Store Mapping',index_col=0)
category_mapping_data = pd.read_excel('../data/Data.xlsx',sheet_name='Category Mapping',index_col=0)
ping_data = pd.read_excel('../data/Data.xlsx',sheet_name='Ping_Information',index_col=0)

# Storing the dataframes as matrices also
demographics_matrix = demographics_data.as_matrix() 
store_mapping_matrix = store_mapping_data.as_matrix()
category_mapping_matrix = category_mapping_data.as_matrix()
ping_matrix = ping_data.as_matrix()

# For each shopper's location getting nearest store and it's category
shopperID_to_nearestCat = {}
shopperID_to_nearestStore = {}
count = 1
for i, row in ping_data.iterrows():
	floor = row['floor']
	person_id = row['Shopper_ID']
	if person_id not in shopperID_to_nearestStore:
		shopperID_to_nearestCat[person_id] = []
		shopperID_to_nearestStore[person_id] = []
	person_lat,person_long = row['lat'], row['lng']
	floor_data = store_mapping_data.loc[store_mapping_data['Floor_Index'] == getFloor_id(floor)]
	dist = []
	for j, shop in floor_data.iterrows():
		shop_lat, shop_long = shop['latitude'], shop['longitude']
		dist.append(haversine(shop_long,shop_lat,person_long,person_lat))
	shop = floor_data.iloc[dist.index(min(dist))]['Store_Name']
	category  = getShop_category(shop)
	shopperID_to_nearestStore[person_id].append(shop)
	shopperID_to_nearestCat[person_id].append(category)
	print count, person_id, '---->', shop, category
	count += 1

ping_shop_freq = {}
ping_cat_freq = {}
count = 1
for key, value in shopperID_to_nearestStore.iteritems():
	for shop in value:
		if shop not in ping_shop_freq:
			ping_shop_freq[shop] = 1
		else:
			ping_shop_freq[shop] += 1
	
	for cat in shopperID_to_nearestCat[key]:
		if cat not in ping_cat_freq:
			ping_cat_freq[cat] = 1
		else:
			ping_cat_freq[cat] += 1
		
	print count, key
	count += 1

ping_shop_freq = sorted(ping_shop_freq.items(), key=lambda x:x[1])
ping_cat_freq  = sorted(ping_cat_freq.items(), key=lambda x:x[1])