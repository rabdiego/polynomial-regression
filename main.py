# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Importing and preparing the data
data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Preparing the features to a polynomial regression
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)

# Training the features with Linear Regression
regressor = LinearRegression()
regressor.fit(X_poly, y_train)

# Creating a range to facilitate the function visualization
X_grid = np.arange(min(X_test), max(X_test), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

# Predicting the results
y_pred = regressor.predict(poly_reg.fit_transform(X_grid))

# Plotting the results
plt.title('Position x Salary', c='m')
plt.xlabel('Position', c='m')
plt.ylabel('Salary', c='m')
plt.scatter(X_test, y_test, c='c')
plt.plot(X_grid, y_pred, c='m')
plt.legend(['Real value', 'Predicted value'])
plt.savefig('plot.png')