from detect import *
from FaceRecognizer import *
import glob

data_dir = '../recognition/orl_faces/'

fr = FaceRecognizer()

for directory in glob.glob(data_dir+'*')[0:2] : #[0:(len(glob.glob(data_dir+'*'))/10)] :
	print directory
	person_name = directory.split('/')[-1]

	for imfile in glob.glob(directory+'/*') :
		print imfile
		im_id = int(imfile.split('/')[-1].split('.')[0])
		if (im_id <= 6) :
			print im_id
			img, score = face_detect(imfile)
			fr.add_face(img, person_name)

print 'FaceRecognizer -- Initialized with training data!'
fr.train()

TEST_SIZE = 0
CORRECT = 0.0

for directory in glob.glob(data_dir+'*')[0:2] : #[0:(len(glob.glob(data_dir+'*'))/10)] :
	person_name = directory.split('/')[-1]

	for imfile in glob.glob(directory+'/*') :
		im_id = int(imfile.split('/')[-1].split('.')[0])
		if (im_id > 6) :
			TEST_SIZE += 1
			img, score = face_detect(imfile)
			guess, confidence = fr.predict(img)
			if (guess == person_name) :
				CORRECT += 1


print 'Accuracy = %g' %( (CORRECT/TEST_SIZE))