"""
Working with the drone dataset
@author: robert brandl
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

"""First attempt at linear regression using wind_speed and velocity_y as
the variables --- not a perfect fit linear regression"""

'import data'
data = pd.read_csv('flights.csv')
print(data.columns)

'visualize x and y variables'
data.plot.scatter('wind_speed', 'velocity_y')

'create training and testing data'
X_train, X_test, Y_train, Y_test = train_test_split(data.wind_speed, data.velocity_y, test_size = 0.2)

'set linear regression variable'
regr = LinearRegression()

'convert pandas array into numpy array using variable from above'
regr.fit(np.array(X_train).reshape(-1,1), Y_train)

'create a set of prediction data using the test data from above'
preds = regr.predict(np.array(X_test).reshape(-1,1))

print(Y_test.head())

print(preds)

'find the difference between the actual results and the predicted results'
residuals = preds - Y_test
print(residuals)

'visualize the error using a histogram'
plt.hist(residuals)

'find the error of the linear regression'
error = mean_squared_error(Y_test, preds) ** 0.5
print(error)


"""second attempt using different variables"""
data.plot.scatter('velocity_x', 'orientation_x')
X_train, X_test, Y_train, Y_test = train_test_split(data.wind_angle, data.linear_acceleration_z, test_size = 0.2)
regr.fit(np.array(X_train).reshape(-1,1), Y_train)
preds = regr.predict(np.array(X_test).reshape(-1,1))
print(Y_test.head())
print(preds)
residuals = preds - Y_test
print(residuals)
plt.hist(residuals)
error = mean_squared_error(Y_test, preds) ** 0.5
print(error)