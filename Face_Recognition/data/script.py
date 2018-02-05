# -*- coding: utf-8 -*-
import urllib
import json
#urllib.urlretrieve("https://firebasestorage.googleapis.com/v0/b/lynkhacksmock.appspot.com/o/10.jpg?alt=media&token=538dec79-1f84-4c47-8185-bfcc3ce4aa3a", "local-filename.jpg")

file = open('dataset.json', 'r') 
labelmap = {}

data = json.load(file)

for elem in data:
	img_name = 'lynk/'+elem['imageuid'] + '.jpg'
	urllib.urlretrieve(elem['url'], img_name)
	labelmap[elem['imageuid']] = elem['name']

with open('labels.json','w') as f :
	f.write(json.dumps(labelmap))
