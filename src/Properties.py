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

# create a figure and axes object
fig, ax = plt.subplots(figsize=(12, 7))
# read in the data and parse by dates (10.2014 - 9.2020)
df_vals = pd.read_csv('property_vals_vs_time.csv', parse_dates=True, index_col='DATE')

# store desired colors in a list
colors = ["#3182bd", "#fc4e2a", "#006d2c", "#cb181d", "#6a51a3",
          "#662506", "#dd1c77", "#525252", "#238b45", "#253494"]

# hold the columns of the dataframe in a list of tuples
cols = df_vals.columns
cols_list = list(enumerate(cols))

# iterate over list and plot each column over daterange
for idx, col in cols_list:
    # create an index variable for the colors list
    idxc = int(idx / 2)
    ax.plot(df_vals.index, df_vals[col], color=colors[idxc]) #, label=col)

# set up an axis formatter for thousands seperators    
y_format = tkr.FuncFormatter(yfunc)
ax.yaxis.set_major_formatter(y_format)

# store indices of flood events
t1 = df_vals.index[24]
t2 = df_vals.index[48]

# format the plot
ax.set_xlabel('Time (months)', size=18, color='k')
ax.set_ylabel('Property Value ($)', size=18, color='k')
ax.set_title('Property Values vs. Time in Flood-Effected Zones', size=25)
ax.axvline(x=t1, color='r', linestyle='--', label='Hurricane Matthew')
ax.axvline(x=t2, color='k', linestyle='--', label='Hurricane Florence')
ax.legend(facecolor="white")
plt.xticks(size=12, rotation=45)
plt.yticks(size=12)
plt.style.use('ggplot')
plt.show()


"""
for col in df.columns:
    ax.plot(df.index, df[col])
plt.show()
"""

"""
for i in range(20):
    idx = int(i / 2)
    #if (max(df[cols[i]]) < 500000):
    ax[idx].plot(df.index, df[cols[i]], color=colors[idx])
    #else:
        #ax2[1, idx].plot(df.index, df[cols[i]], color=colors[idx])
plt.show()
"""

# TODO: make the plot larger
# TODO: annotate flood events on the plot
# TODO: add legend to depict zones
# TODO: change color schemes
# TODO: consider splitting up over multiple plots