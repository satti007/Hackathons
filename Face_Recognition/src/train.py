from detect import *
from FaceRecognizer import *
import glob
import json
import pickle

data_dir = '../data/lynk/'

with open('../data/labels.json') as f :
	label_map = json.load(f)

fr = FaceRecognizer()

count = 1
for file_name in glob.glob(data_dir+'*') :
	print 'Processing -- ',file_name, ' -- ', count
	count += 1
	image_uid = file_name.split('/')[-1].split('.')[0]
	person_name = label_map[image_uid]

	img, score = face_detect(file_name)
	fr.add_face(img, person_name)

print 'FaceRecognizer -- Initialized with training data!'
fr.train()
print 'Done Training'

with open('FRmod_v1.obj', 'w') as filehandler :
	pickle.dump(fr, filehandler)

print 'Saved Model!'