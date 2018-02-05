import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    
class recogDB:
	
	def __init__(self):
		self.K = 4096
		
		self.F_bar = np.array([])
		self.F = {}
		self.weight = {}
		self.name = {}					# index to name

		self.name_inv = {}			# name to index
		self.maxreps = 3 
		self.num_people = 0
		self.threshold = 0.1

		self.score_imp = 1
		self.confidence_imp = 3

	def predict(self, F_, score=1):
		Ffeature = F_
		F_ = F_ / np.linalg.norm(F_)

		F_hat = softmax(np.dot(F_.T, self.F_bar.T))
		print F_hat
		index = np.argmax(F_hat)

		if(F_hat[index] > self.threshold):
			self.add_to_DB(Ffeature, self.name[index], score=score, confidence=F_hat[index])
			return self.name[index], F_hat[index]
		else:
			return ('notfound', 0)

	def add_to_DB(self, F, name, score=1, confidence=1) :
		newbie = name not in self.name.values() 

		if newbie : # New Element
			index = self.num_people
			self.name[index] = name
			self.name_inv[name] = index

			self.F[index] = [F]
			self.weight[index] = [score*self.score_imp + confidence*self.confidence_imp]
			self.num_people += 1

		else :
			index = self.name_inv[name]
			new_weight = score*self.score_imp + confidence*self.confidence_imp

			self.F[index] = [F] + self.F[index][0:(self.maxreps-1)]						  ## Weight Based
			self.weight[index] = [new_weight] + self.weight[index][0:(self.maxreps-1)]


		T = np.zeros(self.K)
		for i in xrange(len(self.weight[index])):
			T = np.subtract(T,-self.weight[index][i]*np.array(self.F[index][i]))

		if newbie :
			try :
				self.F_bar = np.vstack((self.F_bar, T))
			except ValueError :
				self.F_bar = [T]
		else :
			# print T / np.linalg.norm(T)
			# print self.F_bar[index]
			self.F_bar[index] = T / np.linalg.norm(T)
		#print self.F_bar

