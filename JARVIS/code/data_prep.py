import pandas as pd
pd.options.mode.chained_assignment = None 

train_data = pd.read_csv("../data/train.csv",sep ='	')
test_data  = pd.read_csv("../data/test.csv",sep ='	')

train_y = train_data["Runs"]
test_y  = test_data["Runs"]
train_len = train_data.shape[0]

del train_data["Runs"]
del test_data["Runs"]
del train_data["Machine_ID"]
del test_data["Machine_ID"]

frames = [train_data,test_data]
data = pd.concat(frames)

'''
data.isnull().sum(axis=0) ## check for missing values (there are no missing values)
data.dtypes ## Check DataTypes of columns
'''
columns = list(data.columns)

## Delete columns if majority(>98%) rows have same value
for col in columns:
	per = data[col].value_counts(normalize=True)
	# print max(per), col
	if max(per) > 0.98:
		del data[col]

'''
del_columns = set(columns) - set(list(data.columns))
print del_columns
data.shape
'''

## Spliting data back to train and test
train_data = data[:train_len]
test_data = data[train_len:]

train_data["Runs"] = train_y
test_data["Runs"]  = test_y

train_data.to_csv('../data/cleaned_train_data.csv', index=False)
test_data.to_csv('../data/cleaned_test_data.csv', index=False)

