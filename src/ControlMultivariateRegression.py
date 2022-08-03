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

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import linear_model
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn import metrics
import statsmodels.api as sm

# set the working directory
os.chdir("/home/michael/Documents/NCSU/CGA/projects/Urban-Systems-Flooding/data")

# define macros
NUM_OBS = 20
ID = 10021
POST_MATT = 23
POST_FLOR = 48
MTHS_PRE_FLOOD = 99
MESH = 1000
HEIGHT = 10
WIDTH = 10

# create fig and ax objects
fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))

# read in the data, specifying header to be row 0 (10.2016 - 9.2020)
# specify that CUMUL_FLOODS is a categorical variable
#df = pd.read_csv('multivariate_data.csv', dtype={'CUMUL_FLOODS' : 'category'}, index_col = 'PROPERTY_ID', header=0)
df = pd.read_csv('multivariate_control_data.csv', index_col = 'PROPERTY_ID', header=0)

# remove all missing values
df = df.dropna(axis=0)

# tranform appropriate data using a logarithmic transformation
transVars = ['LOT_SIZE', 'SQUARE_FOOTAGE', 'YEAR_BUILT']
#df[transVars] = df[transVars].apply(np.log)

# create a correlation matrix between variables
corrMatrix = df.corr()
# visualize the correlation matrix
sn.heatmap(corrMatrix, annot=True, linewidths=.5, ax=ax)
plt.show()

# create a list of explanatory variables to apply a multiple linear regression
predictors = ['MONTHS_POST_FLOOD', 'YEAR_BUILT',
              'SQUARE_FOOTAGE']
#predictors = ['MONTHS_POST_FLOOD', 'YEAR_BUILT', 'SINGLE_STORY',
#              'SQUARE_FOOTAGE', 'LOT_SIZE', 'CUMUL_FLOODS']
# select variable to apply a simple linear regression 
#predictors = ['MONTHS_POST_FLOOD']

# extract relevant data from dataframes
#Y = df['ABSOLUTE_VALUE']
Y = df['PERCENTAGE_CHANGE_FROM_MEDIAN_PRICE']
X = df[predictors]

# Generate a table of summary statistics for each predictor variable
df_stats = np.round(X.describe(), 2).T
# display the table
print(df_stats)

# with sklearn, use a linear regression model
regr = linear_model.LinearRegression()
# fit the model
regr.fit(X, Y)

# display intercepts and coefficients
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
X = sm.add_constant(X) # adding a constant
 
# with statsmodels, use an ordinary least squares model
model = sm.OLS(Y, X).fit()
# use explanatory variables to predict property values
predictions = model.predict(X) 
 
# obtain and print model summary
print_model = model.summary()
print(print_model)

# Logistic regression model fitting
"""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
"""


"""
# create a linear space equal to number of observations
x = np.linspace(MTHS_PRE_FLOOD, MTHS_PRE_FLOOD + POST_FLOR, MESH)
# define the least squares line
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
    plt.plot(x, y)
    # add features
    plt.xlabel('Months Post Flood', color='k')
    plt.ylabel('Percentage Change From Median Price', color='k')
    plt.title('Percent Change From Median Price vs. Months Post \n\
              (Control Properties)')
    plt.style.use('ggplot')
    plt.legend(facecolor='white')
    plt.show()
"""