# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:19:57 2020

@author: Wes User
"""
#NewThinkTank.com

#http://www.newthinktank.com/2020/08/learn-matplotlib-one-video/


import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import sys


def pause():
    sys.stdout.flush()
    input("Hit Enter to continue")
    

class My_Table():
    def __init__(self, data, col_head, color):
        
        pd.options.display.float_format = '{:,.2f}'.format 

        plt.figure(linewidth=2, tight_layout={'pad':.5}, figsize=(5,3))
        axes_1 = plt.gca()
        axes_1.get_xaxis().set_visible(False)
        axes_1.get_yaxis().set_visible(False)
        plt.box(on=None)
        self.table = plt.table(cellText=data, loc='center', colLabels=col_head,
                     colColours=color)  
        self.table.set_fontsize(18)
        self.table.scale(3,2.5) # width height


#---------------Table 1---------------------

goog_data = pd.read_csv('Goog_NS.csv')
goog_data_np = goog_data.to_numpy()
stk_data = goog_data[-10:] #bottom 10
col_head = goog_data.columns.values  # made generic
stk_data_np = stk_data.to_numpy()
ccolors = plt.cm.Reds(np.full(len(col_head), 0.3))

# create table
google = My_Table(stk_data_np, col_head, ccolors)

plt.show()
pause()

#---------------Table 2---------------------

fruit_data = pd.read_csv('Price.csv') #same data from google file
col_head_f = fruit_data.columns.values  # made generic
prices = col_head_f[1:]
fruit_data.update(fruit_data[prices].applymap('{:,.3f}'.format))
fruit_data_np = fruit_data.to_numpy()
ccolors = plt.cm.YlGn(np.full(len(col_head), 0.7))

# create table
fruit = My_Table(fruit_data_np, col_head_f, ccolors)

#df.update(df[['Age', 'Km']].applymap('{:,.2f}'.format))
#https://stackoverflow.com/questions/53786122/float-format-in-matplotlib-table


#Use the commands below in the console!
#for new windows
#%matplotlib qt5
#for plotting in console
#%matplotblib inline

