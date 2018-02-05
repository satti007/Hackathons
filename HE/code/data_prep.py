import numpy as np
import pandas as pd

test_data = pd.read_csv("../data/test_data.csv")
train_data = pd.read_csv("../data/train_data.csv")

target = train_data["target"]
train_len = train_data.shape[0]

del train_data["target"]
del train_data["connection_id"]
del test_data["connection_id"]

frames = [train_data,test_data]
data = pd.concat(frames)

'''
## check target class
train['target'].value_counts(normalize=True)

## check missing values
data.isnull().sum(axis=0) ## there are no missing values.

## Check DataTypes of columns
data.dtypes

columns = list(data.columns)

## Delete columns if majority(>95%) rows have same value
for col in columns:
	per = data[col].value_counts(normalize=True)
	if max(per) > 0.95:
		del data[col]

del_columns = set(columns) - set(list(data.columns))
data.shape
'''

## Combining levels of categorical data if there are minority(<1%)
columns = list(data.columns)
for col in columns:
	if "cat" in col:
		data[col] = data[col].astype("category")
		per = data[col].value_counts(normalize=True)
		# if max(per) > 0.99:
		# 	# del data[col]
		# else:
		for i in range(0,len(per)):
			if per[per.index[i]] < 0.01:
				data[col].replace(per.index[i],max(per.index.categories)+1,inplace=True)
		data[col] = data[col].astype("category")

## One hot Encoding categorical variables
data = pd.get_dummies(data)

## Spliting data back to train and test
train_data = data[:train_len]
test_data = data[train_len:]

train_data["target"] = target
data = train_data

labels = data['target'].value_counts().index.tolist()
valid_data = pd.DataFrame()
train_data = pd.DataFrame()
for l in labels: # Division in each label
	l_data = data.loc[data['target'] == l]
	data_1, data_2 = np.split(l_data.sample(frac=1), [int(.8*len(l_data))])
	train_data = train_data.append(data_1, ignore_index=True)
	valid_data = valid_data.append(data_2, ignore_index=True)


train_data = train_data.sample(frac=1).reset_index(drop=True)
valid_data = valid_data.sample(frac=1).reset_index(drop=True)

train_data.to_csv('../data/cleaned_train_data_ALL.csv', index=False)
valid_data.to_csv('../data/cleaned_valid_data_ALL.csv', index=False)
test_data.to_csv('../data/cleaned_test_data_ALL.csv', index=False)
