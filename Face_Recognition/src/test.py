from detect import *
import facial_feature_extract_alexnet as ANF
import facial_feature_extract_inceptionv3 as INF
from recogDB import *
from classifyDB import *

# db = recogDB()
db_alex = classifyDB(ANF.FEATURE_SIZE)
db_inv3 = classifyDB(INF.FEATURE_SIZE)

import glob

data_dir = '../recognition/orl_faces/'

for directory in glob.glob(data_dir+'*')[0:5] : #[0:(len(glob.glob(data_dir+'*'))/10)] :
	print directory
	person_name = directory.split('/')[-1]

	for imfile in glob.glob(directory+'/*') :
		print imfile
		im_id = int(imfile.split('/')[-1].split('.')[0])
		if (im_id <= 6) :
			print im_id
			img, score = face_detect(imfile)
			feature = ANF.face_feature_extract(img)
			db_alex.add_to_DB(feature, person_name, score=score, confidence=1)
			feature = INF.face_feature_extract(img)
			db_inv3.add_to_DB(feature, person_name, score=score, confidence=1)

print 'ClassifyDB -- Initialized with all personel!'
db_alex.train()
db_inv3.train()

TEST_SIZE = 0
CORRECT_ALEX  = 0.0
CORRECT_INV3  = 0.0

CORRECT = 0.0

for directory in glob.glob(data_dir+'*')[0:5] : #[0:(len(glob.glob(data_dir+'*'))/10)] :
	person_name = directory.split('/')[-1]

	for imfile in glob.glob(directory+'/*') :
		im_id = int(imfile.split('/')[-1].split('.')[0])
		if (im_id > 6) :
			TEST_SIZE += 1
			img, score = face_detect(imfile)
			
			feature = ANF.face_feature_extract(img)
			guess_alex, confidence_alex = db_alex.predict(feature, score=score)
			if guess_alex == person_name :
				CORRECT_ALEX += 1
			
			feature = INF.face_feature_extract(img)
			guess_inv3, confidence_inv3 = db_inv3.predict(feature, score=score)
			if guess_inv3 == person_name :
				CORRECT_INV3 += 1

			print ('%s_%d -> ALEX : %s [%g] || INV3 : %s [%g]' %(person_name, im_id, guess_alex, confidence_alex, guess_inv3, confidence_inv3))

			if confidence_inv3 > confidence_alex :
				model_guess = guess_inv3
			else :
				model_guess = guess_alex

			if (model_guess == person_name) :
				CORRECT += 1


print 'Accuracy = %g || %g || TRUE- %g' %( (CORRECT_ALEX / TEST_SIZE), (CORRECT_INV3 / TEST_SIZE), (CORRECT/TEST_SIZE))