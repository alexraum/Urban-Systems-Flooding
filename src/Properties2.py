# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 23:29:54 2020

@author: alex raum
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr


# define a helper function to help format y-axis
def yfunc(y, pos):
    s = '{:0,d}'.format(int(y))
    return s

# define plot_data function that plots the data in both lists on
# the same figure in the specified color
def plot_data(daterange, list1, list2, linecolor, fig_width,
              fig_height, label_size, title_size, ticks_size, zone):
    
    # create a figure and axes object
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
     # set up an axis formatter for thousands seperators    
    y_format = tkr.FuncFormatter(yfunc)
    ax.yaxis.set_major_formatter(y_format)
    
    # store indices of flood events
    t1 = daterange[24]
    t2 = daterange[48]
    
    # plot the data
    ax.plot(daterange, list1, color=linecolor)
    ax.plot(daterange, list2, color=linecolor)
    
    # format the plot
    ax.set_xlabel('Time (months)', size=label_size, color='k')
    ax.set_ylabel('Property Value ($)', size=label_size, color='k')
    ax.set_title('Property Values vs. Time in Zone ' + zone, size=title_size)
    ax.axvline(x=t1, color='r', linestyle='--', label='Hurricane Matthew')
    ax.axvline(x=t2, color='k', linestyle='--', label='Hurricane Florence')
    ax.legend(facecolor="white")
    plt.xticks(size=ticks_size, rotation=45)
    plt.yticks(size=ticks_size)
    plt.style.use('ggplot')
    plt.show()


#####################################################################################
    
# define macros
num_zones = 10
fig_width = 10
fig_height = 5
label_size = 16
title_size = 22
ticks_size = 12
    
# read in the data and parse by dates (10.2014 - 9.2020)
df_vals = pd.read_csv('property_vals_vs_time.csv', parse_dates=True, index_col='DATE')

# store desired colors in a list
colors = ["#3182bd", "#fc4e2a", "#006d2c", "#cb181d", "#6a51a3",
          "#662506", "#dd1c77", "#525252", "#238b45", "#253494"]

# store the daterange of the data
dates = df_vals.index
      
# hold the columns of the dataframe in a list of tuples
cols = df_vals.columns
cols_list = list(cols)

# for each of 10 pairs of columns, plot both columns
# in the pair together on their own plot
for num in range(num_zones):
    # store function parameters
    idx1 = (2 * num)
    idx2 = (2 * num + 1)
    vals1 = df_vals[cols_list[idx1]]
    vals2= df_vals[cols_list[idx2]]
    color = colors[num]
    zone = str(num + 1)
    
    # pass the plot data function both columns in the pair
    plot_data(dates, vals1, vals2, color, fig_width, fig_height,
              label_size, title_size, ticks_size, zone)
    