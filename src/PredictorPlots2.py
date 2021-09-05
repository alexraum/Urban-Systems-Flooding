# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 13:46:56 2021

@author: alex raum

This PredictorPlots2 file reads data from both flooded and control properties, where
the "months_post_flood" predictor for control properties and flood properties runs
from 1 to 48 months.

After reading in the data and storing in dataframes, PredictorPlots creates a
visualization of the "percentage_change_from_median_price" response variable 
plotted against the "months_post_flood" predictor variable, where each value of
"percentage_change_from_median_price" is an average over all observations in the
dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# define macros
HEIGHT = 10
WIDTH = 14
MONTHS = 48
NUM_OBS = 20
FLOR = 23
LABEL_SIZE = 26
TITLE_SIZE = 30
TICKS_SIZE = 22
MARK_SIZE = 150

# create a figure and axes object
fig, ax = plt.subplots(figsize = (WIDTH, HEIGHT))

# read the flood data into a dataframe and index by PROPERTY_ID
df_flood = pd.read_csv('multivariate_data2.csv', index_col = 'PROPERTY_ID', header = 0)

# read the control data into a dataframe and index by PROPERTY_ID
df_control = pd.read_csv('multivariate_control_data3.csv', index_col = 'PROPERTY_ID', header = 0)

# extract the response variable (percent change from median price)
Y_flood = df_flood['PERCENTAGE_CHANGE_FROM_MEDIAN_PRICE']
Y_control = df_control['PERCENTAGE_CHANGE_FROM_MEDIAN_PRICE'] 

# extract the predictor variable (months post flood, 1-48)
X_flood = df_flood['MONTHS_POST_FLOOD'].iloc[0:MONTHS]
X_control = df_control['MONTHS_POST_FLOOD'].iloc[0:MONTHS]

# grab the response variable for first observation
avg_Y_flood = np.array(Y_flood.iloc[0:MONTHS])
avg_Y_control = np.array(Y_control.iloc[0:MONTHS])

# define the number of iterations
iters = int(len(Y_flood) / len(avg_Y_flood))

# compute an element-wise average (grouped by property id) on each observation in the response column
for idx in range(1, iters):
    # store the index of the lower bound of the current observation
    lower = idx*MONTHS
    # store the index of the upper bound of the current observation
    upper = (idx + 1)*MONTHS
    # update the response variable by adding the current observation
    avg_Y_flood += np.array(Y_flood.iloc[lower:upper])
    avg_Y_control += np.array(Y_control.iloc[lower:upper])

# scale the sum by the number of observations
avg_Y_flood /= NUM_OBS
avg_Y_control /= NUM_OBS
    
# do the same thing using a list comprehension
flood_vals = [np.array(Y_flood.iloc[i*MONTHS : (i + 1)*MONTHS]) for i in range(iters)]
control_vals = [np.array(Y_control.iloc[i*MONTHS : (i + 1)*MONTHS]) for i in range(iters)]
flood_vals_sum = sum(flood_vals)
control_vals_sum = sum(control_vals)

# scale by the number of observations
mean_flood_vals = flood_vals_sum / NUM_OBS
mean_control_vals = control_vals_sum / NUM_OBS



##############################################################################
###                           VISUALIZATION                                ###

# plot average percentage change from median price in the months following
# Matthew for flooded properties (green)
x_matt_flood = X_flood[0:MONTHS]
y_matt_flood = avg_Y_flood[0:MONTHS]
ax.scatter(x_matt_flood, y_matt_flood, label='flooded properties',
           s=MARK_SIZE , color="#238b45")

# plot average percentage change from median price in the months following
# Matthew for control properties (orange)
x_matt_control = X_control[0:MONTHS]
y_matt_control = avg_Y_control[0:MONTHS]
ax.scatter(x_matt_control, y_matt_control, label='control properties',
           s=MARK_SIZE , color="#fc4e2a")

# add features to the plot
plt.xlabel('Months Post Flood', size=LABEL_SIZE, color='k')
plt.ylabel('Percentage Change From Median Price', size=LABEL_SIZE, color='k')
plt.title('Average Percent Change From Median Price vs.\n Months Post Flood',
          size=TITLE_SIZE)
plt.axvline(x=0, color='r', linestyle='--', label='Hurricane Matthew')
plt.axvline(x=FLOR, color='k', linestyle='--', label='Hurricane Florence')
plt.style.use('ggplot')
plt.legend(facecolor='white', prop={'size': 20})
plt.xticks(size=TICKS_SIZE)
plt.yticks(size=TICKS_SIZE)
plt.show()
