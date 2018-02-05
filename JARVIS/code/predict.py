import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

train_data = pd.read_csv("../data/cleaned_train_data.csv")
test_data  = pd.read_csv("../data/cleaned_test_data.csv")

regr = linear_model.LinearRegression()
features = train_data.shape[1] - 1

regr.fit(train_data.iloc[:,0:features],train_data['Runs'])
test_pred = regr.predict(test_data.iloc[:,0:features])

rmse = np.sqrt(mean_squared_error(test_data['Runs'], test_pred)*test_data.shape[0])/test_data.shape[0]

print('Root mean squared error: %.4f'% rmse)
print('Variance score: %.2f' % r2_score(test_data['Runs'], test_pred))

predictions = pd.DataFrame()
predictions['Actual Value'] = test_data['Runs']
predictions['Predicted Value'] = test_pred
predictions.to_csv('../data/predictions.csv', index=False)

from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(random_state=42)
regr.fit(train_data.iloc[:,0:features],train_data['Runs'])
test_pred = regr.predict(test_data.iloc[:,0:features])


# import matplotlib.pyplot as plt
# plt.plot(test_data["Runs"],test_pred)

