#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt

import seaborn as sb

import math

#there is currently a bug in panda's scatter_matrix that produces a warning that has no affect on what I'm doin :)
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#load in the data and make sure it's correct
df = pd.read_csv('./housing.csv', header=None, delim_whitespace=True, 
                 names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])
display(df.head())
display(df.info())


# In[3]:


#clean data
dep_var = 'MEDV'
display(df.isnull().sum()) #no null values, yay!
display(df.head())


# In[4]:


#standardize data
df = (df - df.mean()) / df.std()
display(df.head())


# In[5]:


#scatterplot matrix
scatter_matrix(df, alpha=.2, figsize=(8,8), diagonal='kde')
plt.show()


# In[6]:


#correlation matrix
corr_matrix = df.corr()
display(corr_matrix)


# In[7]:


#heatmap
heatmap = sb.heatmap(corr_matrix)
plt.show()


# In[8]:


#residual plot
def res_plot(x_):
    y_ = dep_var
    sb.residplot(x=x_,y=y_, data=corr_matrix, lowess=True)
    plt.show()
    
corr_matrix.apply(res_plot)


# In[9]:


#independent vars based on heatmap correlation to MEDV
ind_var_count = 3
ind_var_1 = 'LSTAT'
ind_var_2 = 'RM'
ind_var_3 = 'PTRATIO'


# In[10]:


#split the data: 75% train, 25% test
shuffled_df = df[[ind_var_1, ind_var_2, ind_var_3, dep_var]].sample(frac=1)
train_size = int(0.75 * len(shuffled_df))

train_df = shuffled_df[:train_size]
test_df = shuffled_df[train_size:]

display(train_df.head())
display(test_df.head())
display(train_df.info())
display(test_df.info())


# In[11]:


#coefficients in order: b0, LSTAT, RM, PTRATIO
x = train_df.iloc[:,0:3]
y = train_df.iloc[:,3]
x['b0'] = 1
x = x.ix[:, ['b0', ind_var_1, ind_var_2, ind_var_3]]
display(x.head())
display(y.head())

coeff = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
display(coeff)

print ("Y = ", float(coeff[0]), " + ", float(coeff[1]), "LSTAT + ", float(coeff[2]), "RM + ", float(coeff[3]), "PTRATIO")

#apply the regression model equation
train_df[ind_var_1] = train_df[ind_var_1].apply(lambda x: x*float(coeff[1]))
train_df[ind_var_2] = train_df[ind_var_2].apply(lambda x: x*float(coeff[2]))
train_df[ind_var_3] = train_df[ind_var_3].apply(lambda x: x*float(coeff[3]))

train_df['predy'] = 0

for i in range(len(train_df)):
    train_df.iloc[i,4] = float(coeff[0]) + train_df.iloc[i,0] + train_df.iloc[i,1] + train_df.iloc[i,2]
    
display(train_df.head())


# In[16]:


#r squared
train_df['r^2'] = 0
for i in range(len(train_df)):
    train_df.iloc[i,5] = pow(train_df.iloc[i,3] - train_df.iloc[i,4], 2)
    
display(train_df.head())


# In[17]:


#adjusted r squared
train_df['adj r^2'] = 0
length = len(train_df)
for i in range(len(train_df)):
    train_df.iloc[i,6] = 1 - (((1-train_df.iloc[i,5]) * (length - 1)) / (length - 3 - 1))
    
display(train_df.head())


# In[18]:


#RMSE
rmse = math.sqrt(train_df['r^2'].sum(axis=0) / length)
print("RMSE: ", rmse)

#RMSE is in the middle, meaning there is a 50% error between the data and the regression line 
#on average


# In[19]:


#I'm not sure what to do with the testing data at this point. I'm assuming we'd want to adjust the coefficients
#based on the value of adjusted r, but I don't know how


# In[ ]:




