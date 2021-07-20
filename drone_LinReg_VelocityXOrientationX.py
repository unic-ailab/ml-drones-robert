
"""
@author: robert brandl
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

'second attempt at linear regression proved more successful, with less error'

'import data'
data = pd.read_csv('flights.csv')
print(data.columns)

'visualize x and y variables'
data.plot.scatter('velocity_x', 'orientation_x')

'split into testing and training data'
X_train, X_test, Y_train, Y_test = train_test_split(data.velocity_x, data.orientation_x, test_size = 0.2)

'set a linear regression variable'
regr = LinearRegression()

'convert pandas array into numpy array using variable from above'
regr.fit(np.array(X_train).reshape(-1,1), Y_train)

'create a set of prediction data using the test data from above'
preds = regr.predict(np.array(X_test).reshape(-1,1))

'compare the actual data to the predictions'
print(Y_test.head())
print(preds)

'find the difference between the actual results and the predicted results'
residuals = preds - Y_test
print(residuals)

'visualize the error using a histogram'
plt.hist(residuals)
plt.title('Distribution of Residuals');
plt.xlabel('Error'); 
plt.ylabel('Count')

'find the error of the linear regression'
error = mean_squared_error(Y_test, preds) ** 0.5
print(error)
# error close to 0.05 each run

""" Using the x-velocity and x-orientation for linear regression proved more
accurate, establishing the obvious relationship that changing the velocity
in the x direction affects the orientation of the drone in the x direction.
The high accuracy is demonstrated through the low error, as well as the
majority of error in the histogram comparing the residuals near 0
"""
