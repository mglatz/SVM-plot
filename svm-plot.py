'''
simple script to visualize how sklearn.svm.SVC's parameters C and gamma affects the the learning

based on https://stackoverflow.com/questions/43284811/plot-svm-with-matplotlib
requires mlxtend library
'''

import numpy as np
import pandas as pd
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import random

##################################################################
#various DataSets generated from random numbers

def DataSeparatedWith1Bad():
    #Create arbitrary dataset for example
    ab = [0,2]
    bb = [4,6]
    ax = np.random.uniform(low=ab[0], high=ab[1], size=25)
    bx = np.random.uniform(low=bb[0], high=bb[1], size=25)
    ay = np.random.uniform(low=ab[0], high=ab[1], size=25)
    by = np.random.uniform(low=bb[0], high=bb[1], size=25)

    df1 = pd.DataFrame({'x': ax,
                    'y':  ay,
                    'Category':        [1]*25}
    )
    df2 = pd.DataFrame({'x': bx,
                    'y':  by,
                    'Category':        [2]*25}
    )
    dfx = pd.concat([df1,df2])

    #this moves category 1 intem into category 2 region to have better visualization of what changes in C do
    dfx.iloc[0,1]=5
    dfx.iloc[0,2]=5

    return dfx

def DataRandom():
    #Create arbitrary dataset for example
    ab = [0,6]
    bb = [0,6]
    ax = np.random.uniform(low=ab[0], high=ab[1], size=25)
    bx = np.random.uniform(low=bb[0], high=bb[1], size=25)
    ay = np.random.uniform(low=ab[0], high=ab[1], size=25)
    by = np.random.uniform(low=bb[0], high=bb[1], size=25)

    df1 = pd.DataFrame({'x': ax,
                    'y':  ay,
                    'Category':        [1]*25}
    )
    df2 = pd.DataFrame({'x': bx,
                    'y':  by,
                    'Category':        [2]*25}
    )
    dfx = pd.concat([df1,df2])

    return dfx

def DataCircle():
    #Create arbitrary dataset for example    
    ab = [-2,2]
    bb = [-6,6]
    ax = np.random.uniform(low=ab[0], high=ab[1], size=25)
    bx = np.random.uniform(low=bb[0], high=bb[1], size=25)
    ay = np.random.uniform(low=ab[0], high=ab[1], size=25)
    by = np.random.uniform(low=bb[0], high=bb[1], size=25)

    i=0
    for x in bx:        
        while x >= -2 and x <= 2:
            x = random.randrange(-6,6)
        bx[i]=x
        i=i+1        

    i=0
    for x in by:
        while x >= -2 and x <= 2:
            x = random.randrange(-6,6)
        by[i]=x
        i=i+1
        

    df1 = pd.DataFrame({'x': ax,
                    'y':  ay,
                    'Category':        [1]*25}
    )
    df2 = pd.DataFrame({'x': bx,
                    'y':  by,
                    'Category':        [2]*25}
    )
    dfx = pd.concat([df1,df2])

    return dfx

##################################################################
#select dataset
#uncomment the one to be used

#dfx = DataSeparatedWith1Bad()
#dfx = DataRandom()
dfx=DataCircle()

##################################################################
# Fit Support Vector Machine Classifier
X = dfx[['x', 'y']]
y = dfx['Category']

##################################################################
#values to plot

#Cs = [0.01,0.1,1,10,100,1000]
Cs = [0.1,1,10,1000]
#Cs = [0.01,0.1,1]

Gs = [10,1,0.1,0.01,0.001]
#Gs = [1,0.1,0.001]
##################################################################

#create charts
fig, axes = plt.subplots(nrows=len(Cs),ncols=len(Gs))
r=0
i=0

for c in Cs:
    for g in Gs:        
        
        print(f'calculating SVM for C={c} amd gamma={g}')

        clf = svm.SVC(decision_function_shape='ovr',C=c,gamma=g,kernel='rbf')
        clf.fit(X.values, y.values) 

        # Plot Decision Region using mlxtend's awesome plotting function
        plot_decision_regions(X=X.values, 
                              y=y.values,
                              clf=clf, 
                              legend=0,
                              ax=axes[r][i])
        
        axes[r][i].xaxis.set_visible(False)
        axes[r][i].yaxis.set_visible(False)

        axes[r][i].title.set_text(f'C={c}; g={g}')
        i+=1
    r+=1
    i=0

#set window size to full screen
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.show()