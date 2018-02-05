from math import radians, cos, sin, asin, sqrt

def aversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    
    return c * r

category_mapping_matrix = pd.read_excel('../data/Data.xlsx',sheet_name='Category Mapping',index_col=0).as_matrix()

category_mapping_matrix[np.where(category_mapping_matrix[:,0]=='Shoe Wiz')[0][0]][2]
np.extract(condition,category_mapping_matrix)

list= ['Pasta Moto', 'Kichi Grill', 'M.Y. China', 'Chipotle',
       'Fire of Brazil', 'Teavana', 'Cako Bakery',
       'Andale Mexican Restaurant', 'The Body Shop', 'Amiri Salon', 'Origins']

A=[category_mapping_matrix[np.where(category_mapping_matrix[:,0]==name)[0][0]][2] for name in list ]

for name in list:
	print name
	print np.where(category_mapping_matrix[:,0]==name)[0][0]	


haversine(37.66146302,-122.2986728,37.78442,-122.406832)

def cosine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    theta = sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(dlon) 
    c = acos(theta)
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    
    return c * r

cosine(37.66146302,-122.2986728,37.78442,-122.406832)

d = acos( sin φ1 ⋅ sin φ2 + cos φ1 ⋅ cos φ2 ⋅ cos Δλ ) ⋅ R

sqrt(46000)