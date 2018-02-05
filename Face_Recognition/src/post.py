from detect import *
from FaceRecognizer import *
import glob
import pickle
import requests
import json

data_dir = '../data/lynk/'

verify_url = 'https://us-central1-lynkhacksmock.cloudfunctions.net/verifyface'
team_name = 'NADS'

output_json = []

with open('FRmod_v1.obj', 'r') as filehandler :
 fr = pickle.load(filehandler) 

log_response = open('response.log', 'w')

count = 1
for file_name in glob.glob(data_dir+'*') :
	print 'Predicting -- ',file_name, ' -- ', count
	count += 1
	image_uid = file_name.split('/')[-1].split('.')[0]
	# person_name = label_map[image_uid]

	img, score = face_detect(file_name)
	guess, confidence = fr.predict(img)

	data = {"teamname":team_name, "imageuid": image_uid, "name":guess}

	output_json.append(data)
	# r = requests.post(verify_url, data)
	# log_response.write(r.text + '\n')

with open('output.json','w') as f :
	f.write(json.dumps(output_json))
# print output_json

log_response.close()
