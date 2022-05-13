# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 00:25:55 2021

@author: alex raum
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# set the working directory
os.chdir("/home/michael/Documents/NCSU/CGA/projects/Urban-Systems-Flooding/data")

# define macros
HEIGHT = 10
WIDTH = 14
MONTHS = 48
NUM_OBS = 20
POST_MATT = 23
POST_FLOR = 48
LABEL_SIZE = 26
TITLE_SIZE = 30
TICKS_SIZE = 22
MARK_SIZE = 150

# create a figure and axes object
fig, ax = plt.subplots(figsize = (WIDTH, HEIGHT))

# read the data into a dataframe and index by PROPERTY_ID, store results in a data frame
df = pd.read_csv('multivariate_control_data.csv', index_col = 'PROPERTY_ID', header = 0)

# extract the response variable (percent change from median price)
Y = df['PERCENTAGE_CHANGE_FROM_MEDIAN_PRICE']

# extract the predictor variable (months post flood, 1-23 and 1-25)
X = df['MONTHS_POST_FLOOD'].iloc[0:MONTHS]

# grab the response variable for first observation
avg_Y = np.array(Y.iloc[0:MONTHS])

# dynamically define the number of iterations
iters = int(len(Y) / len(avg_Y))

# compute an element-wise average (grouped by property id) on each observation in the response column
for idx in range(1, iters):
    # store the index of the lower bound of the current observation
    lower = idx*MONTHS
    # store the index of the upper bound of the current observation
    upper = (idx + 1)*MONTHS
    # update the response variable by adding the current observation
    avg_Y += np.array(Y.iloc[lower:upper])

# scale the sum by the number of observations
avg_Y /= NUM_OBS
    
# achieve the same result by using a list comprehension
vals = [np.array(Y.iloc[i*MONTHS : (i + 1)*MONTHS]) for i in range(iters)]
val_sum = sum(vals)

# scale by the number of observations
mean_vals = val_sum / NUM_OBS 


##############################################################################
###                          VISUALIZATION                                 ###

# plot average percentage change in the months following Matthew (pink)
x_matt = X[0:POST_MATT]; y_matt = avg_Y[0:POST_MATT]
ax.scatter(x_matt, y_matt, label='post-Matthew', s=MARK_SIZE , color="#dd1c77")

# plot average percentage change in the months following Florence (green)
x_flor = X[POST_MATT:POST_FLOR]; y_flor = avg_Y[POST_MATT:POST_FLOR]
ax.scatter(x_flor, y_flor, label='post-Florence', s=MARK_SIZE, color="#238b45")

# add features to the plot
plt.xlabel('Months Post Flood', size=LABEL_SIZE, color='k')
plt.ylabel('Percentage Change From Median Price', size=LABEL_SIZE, color='k')
plt.title('Average Percent Change From Median Price vs.\n Months Post Flood (Control Properties)',
          size=TITLE_SIZE)
plt.style.use('ggplot')
plt.legend(facecolor='white', prop={'size': 20})
plt.xticks(size=TICKS_SIZE)
plt.yticks(size=TICKS_SIZE)
plt.show()
