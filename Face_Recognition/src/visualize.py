import sys

from detect import *
from FaceRecognizer import *
import pickle
import cv2

with open('FRmod_v1.obj', 'r') as filehandler :
 fr = pickle.load(filehandler) 


if (len(sys.argv) != 2) :
	exit()
else :
	file_name = sys.argv[1]

	img, score = face_detect(file_name)

	guess, confidence = fr.predict(img)
	print confidence

	import matplotlib.pyplot as plt
	import matplotlib.image as mpimg

	fig = plt.figure()
	aux = cv2.imread(file_name)
	ax1 = fig.add_subplot(1,2,1)
	ax1.imshow(aux)

	aux = img
	ax2 = fig.add_subplot(1,2,2)
	ax2.imshow(aux)
	plt.title(guess)
	plt.show()