'Robert Brandl --- Drone Dataset completed regression analysis using flights.csv'

# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# No warnings about setting value on copy of slices (if necessary)
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 60)

# Matplotlib for visualization
import matplotlib.pyplot as plt

# Set default font size
plt.rcParams['font.size'] = 24

#more visualization tools
from IPython.core.pylabtools import figsize

# Using Seaborn library for visualization
import seaborn as sns
sns.set(font_scale = 2)

# Imputing missing values and scaling values --- preprocessing
from sklearn.preprocessing import  MinMaxScaler
from sklearn.impute import SimpleImputer

# Machine Learning Models (4)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

#creating training and testing data
from sklearn.model_selection import train_test_split


# Read in data into a dataframe (pandas)
data = pd.read_csv('flights.csv')

# Display top of dataframe
print(data.head())

# See the column data types and non-missing values
print(data.info())

#using the wind speed variable, seek out correlations
correlations_data = data.corr()['wind_speed'].sort_values()
print(correlations_data)

'FOUND: correlation between wind speed and velocity y'


# Split into 70% training and 30% testing set
X, X_test, y, y_test = train_test_split(data.wind_speed, data.velocity_y, 
                                        test_size = 0.3, 
                                        random_state = 42)

# Function to calculate mean absolute error (using numpy)
def mae(y, y_pred):
    return np.mean(abs(y - y_pred))

#finding a baseline level of error using the median of the y values
baseline_guess = np.median(y)
print('The baseline guess is a score of %0.2f' % baseline_guess)
print("Baseline Performance on the test set: MAE = %0.4f" % mae(y_test, baseline_guess))

'Preprocessing of data'
# Create an imputer object with a median filling strategy
imputer = SimpleImputer(strategy='median')

# Train on the training features
imputer.fit(np.array(X).reshape(-1,1))

# Transform both training data and testing data
X = imputer.transform(np.array(X).reshape(-1,1))
X_test = imputer.transform(np.array(X_test).reshape(-1,1))


# Create the scaler object with a range of 0-1
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit on the training data
scaler.fit(X)

# Transform both the training and testing data
X = scaler.transform(X)
X_test = scaler.transform(X_test)

# Convert y to one-dimensional array (vector)
y = np.array(y).reshape((-1, ))
y_test = np.array(y_test).reshape((-1, ))

'Start of regression analysis'
# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model):
    
    # Train the model
    model.fit(X, y)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)
    
    # Return the performance metric
    return model_mae

lr = LinearRegression()
lr_mae = fit_and_evaluate(lr)

print('Linear Regression Performance on the test set: MAE = %0.4f' % lr_mae)

random_forest = RandomForestRegressor(random_state=60)
random_forest_mae = fit_and_evaluate(random_forest)

print('Random Forest Regression Performance on the test set: MAE = %0.4f' % random_forest_mae)

gradient_boosted = GradientBoostingRegressor(random_state=60)
gradient_boosted_mae = fit_and_evaluate(gradient_boosted)

print('Gradient Boosted Regression Performance on the test set: MAE = %0.4f' % gradient_boosted_mae)

knn = KNeighborsRegressor(n_neighbors=10)
knn_mae = fit_and_evaluate(knn)

print('K-Nearest Neighbors Regression Performance on the test set: MAE = %0.4f' % knn_mae)

#setting visualization style
plt.style.use('fivethirtyeight')
figsize(8, 6)

# Dataframe to hold the results
model_comparison = pd.DataFrame({'model': ['Linear Regression', 
                                           'Random Forest', 'Gradient Boosted',
                                            'K-Nearest Neighbors'],
                                 'mae': [lr_mae,  random_forest_mae, 
                                         gradient_boosted_mae, knn_mae]})

# Horizontal bar chart of test mae
model_comparison.sort_values('mae', ascending = False).plot(x = 'model', y = 'mae', kind = 'barh',
                                                           color = 'red', edgecolor = 'black')

# Plot formatting
plt.ylabel(''); 
plt.yticks(size = 14); 
plt.xlabel('Mean Absolute Error'); 
plt.xticks(size = 14)
plt.title('Model Comparison on Test MAE', size = 20);

'BEST MODEL: Random Forest Regressor'

#use the best model to generate new predictions
plt.clf()
preds = random_forest.predict(np.array(X_test).reshape(-1,1))
figsize(8, 8)

# Density plot of the final predictions and the test values
sns.kdeplot(preds, label = 'Predictions')
sns.kdeplot(y_test, label = 'Values')

# Label the plot
plt.xlabel('Wind Speed'); 
plt.ylabel('Velocity-Y');
plt.title('Test Values and Predictions');

figsize = (6, 6)

# Calculate the residuals 
residuals = preds - y_test

# Plot the residuals in a histogram
plt.hist(residuals, color = 'red', bins = 20,
         edgecolor = 'black')
plt.xlabel('Error'); 
plt.ylabel('Count')
plt.title('Distribution of Residuals');