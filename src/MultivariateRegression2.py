#  -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 22:11:42 2020

@author: alex raum
"""

# TODO: Consider creating a correlation matrix between variables, all correlation
#       factors between variables should be < 0.5
# TODO: Evaluate the accuracy of the model (model validation)
# TODO: For effective model validation, exclude some data from the model
#       building process, and then use those to test model's accuracy on
#       data it hasn't seen before

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn import metrics
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm

# define macros
NUM_OBS = 20
ID = 10001
POST_MATT = 23
POST_FLOR = 48
MESH = 1000
HEIGHT = 10
WIDTH = 10
FILE = 'multivariate_data.csv'

# create fig and ax objects
fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))

# read in the data, specifying header to be row 0 (10.2016 - 9.2020)
# specify that CUMUL_FLOODS is a categorical variable
#df = pd.read_csv('multivariate_data.csv', dtype={'CUMUL_FLOODS' : 'category'}, index_col = 'PROPERTY_ID', header=0)
df = pd.read_csv(FILE, index_col = 'PROPERTY_ID', header=0)

# remove all missing values
df = df.dropna(axis=0)

# tranform appropriate data using a logarithmic transformation
transVars = ['LOT_SIZE', 'SQUARE_FOOTAGE', 'YEAR_BUILT']
df[transVars] = df[transVars].apply(np.log)

# create a correlation matrix between variables
corrMatrix = df.corr()
# visualize the correlation matrix
sn.heatmap(corrMatrix, annot=True, linewidths=.5, ax=ax)
plt.show()
  
# create a list of explanatory variables to apply a multiple linear regression
predictors = ['MONTHS_POST_FLOOD', 'YEAR_BUILT', 'SINGLE_STORY',
              'SQUARE_FOOTAGE', 'LOT_SIZE', 'CUMUL_FLOODS']
# select variable to apply a simple linear regression 
#predictors = ['MONTHS_POST_FLOOD']

# extract relevant data from dataframes
#y1 = df['ABSOLUTE_VALUE']
y1 = df['PERCENTAGE_CHANGE_FROM_MEDIAN_PRICE']
X1 = df[predictors]

# Generate a table of summary statistics for each predictor variable
df_stats = np.round(X1.describe(), 2).T
# display the table
print(df_stats)

# with sklearn, use a linear regression model
regr = linear_model.LinearRegression()
# fit the model
regr.fit(X1, y1)

# display intercepts and coefficients
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# extract relevant data from dataframes
y2 = df['ABSOLUTE_VALUE']
X2 = df[predictors]

# Split the data, using some data as training to fit the model and the
# rest as validation data
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, random_state=0)
#logreg = LogisticRegression()
#logreg.fit(X_train, y_train)

# with statsmodels
#X2_train = sm.add_constant(X2_train) # adding a constant
 
# with statsmodels, use an ordinary least squares model
# for comparison
model = sm.OLS(y1, X1).fit()
#model = sm.OLS(y2_train, X2_train).fit()
# use explanatory variables to predict property values
predictions = model.predict(X2_test) 
 
# obtain and print model summary
print_model = model.summary()
print(print_model)

# calculate the mean absolute error
mae = mean_absolute_error(y2_test, predictions)
print('\nThe mean absolute error between predicted and observed is', mae.round(2))

"""
x = np.linspace(0, (POST_FLOR - POST_MATT), MESH)
y = regr.coef_ * x + regr.intercept_


# preliminary analysis: 
# check for a linear relationship between variables
for idx in range(NUM_OBS):
    df_obs = df.loc[ID + idx]
    # grab the first 23 months (after Matthew)
    df_matt = df_obs[0 : POST_MATT]
    # grab the next 25 months (after Florence)
    df_flor = df_obs[POST_MATT : POST_FLOR]
    # plot post-Matthew months in red
    plt.scatter(df_matt['MONTHS_POST_FLOOD'], df_matt['PERCENTAGE_CHANGE_FROM_MEDIAN_PRICE'],
                label='post-Matthew', color='r')
    # plot post-Florence months in blue
    plt.scatter(df_flor['MONTHS_POST_FLOOD'], df_flor['PERCENTAGE_CHANGE_FROM_MEDIAN_PRICE'],
                label='post-Florence', color='b')
    plt.plot(x, y, color='k', label='Regression Line')
    # add features
    plt.xlabel('Months Post Flood', color='k')
    plt.ylabel('Percentage Change From Median Price', color='k')
    plt.title('Percent Change From Median Price vs. \n Months Post Flood')
    plt.style.use('ggplot')
    plt.legend(facecolor='white')
    plt.show()
"""