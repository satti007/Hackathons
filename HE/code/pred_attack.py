import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# def svc_param_selection(X, y, nfolds):
#     costs  = [0.1, 1, 10, 50]
#     gammas = [0.01, 0.1, 1]
#     param_grid = {'C': costs, 'gamma' : gammas}
#     grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
#     grid_search.fit(X, y)
#     grid_search.best_params_
#     return grid_search.best_params_

test_data = pd.read_csv('../data/cleaned_test_data_ALL.csv')
valid_data = pd.read_csv('../data/cleaned_valid_data_ALL.csv')
train_data = pd.read_csv('../data/cleaned_train_data_ALL.csv')

Tcom_class = train_data["target"].replace(to_replace=2, value=0, inplace=False)
Vcom_class = valid_data["target"].replace(to_replace=2, value=0, inplace=False)

features = train_data.shape[1] - 1
T_X,T_y = train_data.iloc[:,0:features],Tcom_class
V_X,V_y = valid_data.iloc[:,0:features],Vcom_class

model = LogisticRegression()
model.fit(T_X,T_y)
acc = model.score(V_X,V_y)

close_Tdata = train_data[train_data.target != 1]
close_Vdata = valid_data[valid_data.target != 1]

T_X,T_y = close_Tdata.iloc[:,0:features],close_Tdata["target"]
V_X,V_y = close_Vdata.iloc[:,0:features],close_Vdata["target"]

model = svm.SVC(C=100)
model.fit(T_X,T_y)
acc = model.score(V_X,V_y)



'''
features = train_data.shape[1] - 1
T_X,T_y = train_data.iloc[:,0:features],train_data["target"]
V_X,V_y = valid_data.iloc[:,0:features],valid_data["target"]
split_feat = ["sqrt","log2",None]
valid_acc = []
file = open("valid_acc.log","a")

for i in range(1,11):
	trees = 100*i
	for feat in split_feat:
		model = RandomForestClassifier(n_estimators=trees,max_features=feat,random_state=0)
		model.fit(T_X,T_y)
		acc = model.score(V_X,V_y)
		valid_acc.append(acc)
		file.write("n_estimators: {}, max_features: {}, acc: {}\n".format(trees,feat,acc))

file.close()


# best_params = svc_param_selection(X,y,3)
# model = svm.SVC()
# model.fit(X,y)
# model.fit(X,y,C = best_params["C"],gamma = best_params["gamma"])

# model = RandomForestClassifier(n_estimators=900,max_features=None,random_state=0)
model = RandomForestClassifier()
model.fit(T_X,T_y)

prediction = model.predict(test_data)
S = pd.Series(prediction)
S.value_counts()

# Submission file
sub = pd.read_csv('../data/sample_submission.csv')
sub['target'] = prediction
sub.to_csv('../data/sub_ALL_RF_900.csv', index=False)
'''