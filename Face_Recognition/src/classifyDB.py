from sklearn import tree
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class classifyDB:
	
	def __init__(self, feature_size):
		self.K = feature_size

		self.X = np.array([])
		self.Y = []
		
		self.name = {}					# index to name

		self.name_inv = {}			# name to index
		self.alpha = 0.9				# Moving average
		self.threshold = 0.16
		self.num_people = 0

	def train(self) :
		# self.clf = tree.DecisionTreeClassifier()
		# self.clf = svm.SVC(probability=True)
		self.clf = RandomForestClassifier(n_estimators=max(10*self.num_people,10), max_depth=5, max_features='sqrt')
		Y = np.array(self.Y)
		Y = Y.reshape(-1,1)

		self.reduction = PCA(n_components = 50)
		X = self.reduction.fit_transform(self.X, Y)

		self.clf = self.clf.fit(X, Y)

	def predict(self, F_, score=1):
		prob = self.clf.predict_proba(self.reduction.transform(F_))[0]

		face_id = np.argmax(prob)
		if (prob[face_id] > self.threshold) :
			return self.name[face_id], prob[face_id]
		else : 
			return ('notfound', 0)

	def add_to_DB(self, F, name, score=1, confidence=1) :
		newbie = name not in self.name.values() 

		if newbie : # New Element
			index = self.num_people
			self.name[index] = name
			self.name_inv[name] = index

			self.num_people += 1

		try :
			self.X = np.vstack((self.X, F))
		except ValueError :
			self.X = [F]

		if newbie :
			self.Y.append(index)
		else :
			self.Y.append(self.name_inv[name])

		# else :
		# 	index = self.name_inv[name]
		# 	T = self.alpha*self.X[index] + (1-self.alpha)*F
		# 	self.X[index] = T

