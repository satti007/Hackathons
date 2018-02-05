from detect import *
import facial_feature_extract_alexnet as ANF
import facial_feature_extract_inceptionv3 as INF
from recogDB import *
from classifyDB import *

class FaceRecognizer :

	def __init__(self) :
		self.db_alex = classifyDB(ANF.FEATURE_SIZE)
		self.db_inv3 = classifyDB(INF.FEATURE_SIZE)

	def add_face(self, img, name) :
		feature = ANF.face_feature_extract(img)
		self.db_alex.add_to_DB(feature, name)

		feature = INF.face_feature_extract(img)
		self.db_inv3.add_to_DB(feature, name)

	def train(self) :
		self.db_alex.train()
		self.db_inv3.train()

	def predict(self, img) :
			feature = ANF.face_feature_extract(img)
			guess_alex, confidence_alex = self.db_alex.predict(feature)
			
			feature = INF.face_feature_extract(img)
			guess_inv3, confidence_inv3 = self.db_inv3.predict(feature)
			
			if confidence_inv3 >= confidence_alex :
				guess = guess_inv3
				confidence = confidence_inv3
			else :
				guess = guess_alex
				confidence = confidence_alex
			return guess, confidence
