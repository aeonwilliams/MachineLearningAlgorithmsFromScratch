#!/usr/bin/env python
# coding: utf-8

# # Gaussian Naive Bayes
# #### Aeon Williams
# #### CS398F2020

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>")) # makes the notebook fill the whole window

from bae0n_utils import CorrMatrixAnalysis

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

from functools import reduce
from operator import mul, itemgetter, add
from math import exp, sqrt, pi, log

from sklearn.model_selection import train_test_split # splitting the dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore') # mpl produces warnings about it's own code, doesn't affect what I'm doing at all


# ## Gaussian Naive Bayes Algorithm

# ### Extra Stuff

# In[2]:


# Since the code behind this model didn't take long, I used 
# it as an opportunity to deepen my understanding of python
# and practice comprehensions. I know this is bad practice when
# other people will be using your code, because of readability,
# so I made a clearer version as well :)
def naive_bayes_one_line(ls, pred_row):
    return max({val: (rows[0][-1]/(rows[0][-1] * len(rows))) * reduce(mul, [(1 / (sqrt(2 * pi) * rows[i][1])) * exp(-((pred_row[i]-rows[i][0])**2 / (2 * rows[i][1]**2 ))) for (i, j) in enumerate(rows)]) for (val, rows) in {val: [(sum(column)/float(len(column)), sqrt(sum([(x-mean(column))**2 for x in column]) / float(len(column)-1)), len(column)) for column in zip(*rows)][:-1] for val, rows in {row[-1]: [x for x in ls if x[-1] == row[-1]] for row in ls}.items()}.items()}
.items(), key=itemgetter(1))[0]


# ### Math Tools

# In[3]:


# Calculate the mean of a list of numbers
def mean(col):
    if len(col) == 0:
        return 0
    return sum(col)/float(len(col))
 
# Calculate the standard deviation of a list of numbers
def stdev(col):
    avg = mean(col)
    length = float(len(col)-1)
    if length == 0:
        return 0
    return sqrt(sum([(x-avg)**2 for x in col]) / length)

def calculate_probability(x, mean, stdev):
    if stdev == 0:
        return 0
    return (1 / (sqrt(2 * pi) * stdev)) * exp(-((x-mean)**2 / (2 * stdev**2 )))


# ### Predicting Class Value

# Laplace:
# 
# <img src="./laplace.png" width=200 height=200 style="float:left" />
# 
# <br><br><br><br>
# 
# Log Probabilities:
# 
# <img src="./log.png" width=400 height=200 style="float:left"/>

# In[4]:


# Uses Gaussian Naive Bayes and a training set to predict the class value for the given test sample
def predict_label(ls, test):
    # Creates a dictionary of {label value, rows from train with that value}
    class_dict = {row[-1]: [x for x in ls if x[-1] == row[-1]] for row in ls}
    # Creates a dictionary of {label value, statistic list of mean, stdev, and total number of rows for the row groups in train that 
    # have that label}. 
    statistics = {val: [(mean(column), stdev(column), len(column)) for column in zip(*rows)][:-1] for val, rows in class_dict.items()}
    # Loops through each label value, row statistic pair.
    # For each label value, start off the probability calculation with number of rows in that statistic group 
    # divided by the total number of rows for that label value. The probability is then addiditively 
    # multiplied by the gaussian probability of the corresponding values in the test sample statistics for the 
    # rows in train that have the same label value. This creates a dict of {label value, probability}, and the
    # highest probability is chosen as the predicted value for the test sample.
    # Implements Laplace smoothing & log probabilities.
    prob = {val: np.log(rows[0][-1]+1/((rows[0][-1]) * len(rows))+len(rows[0])) + sum([np.log(calculate_probability(test[i], rows[i][0], rows[i][1])) for (i, j) in enumerate(rows)]) for (val, rows) in statistics.items()}
    # NO LOG PROB: prob = {val: (rows[0][-1]/(rows[0][-1] * len(rows))) * reduce(mul, [calculate_probability(test[i], rows[i][0], rows[i][1]) for (i, j) in enumerate(rows)]) for (val, rows) in statistics.items()}
    # The highest probability is the most accurate prediction for the test label value
    return max(prob.items(), key=itemgetter(1))[0]


# ### Helper Functions & Analysis Tools

# In[5]:


# Uses the train set to create a list of predictions for each item in the test set
def naive_bayes(train, test):
    return [predict_label(train, row) for row in test]

# Takes a dataframe, splits into train and test, and produces the average accuracy of the model 
# (increased # of runs = better indicator of true accuracy)
def evaluate_model(df, runs=1, algorithm='default', test_size=.4, train_size=.6, return_vals=False):
    accuracies, actuals, preds = [], [], []
    for i in range(runs):
        # My Algorithm
        if algorithm == 'default':
            df_list = df.iloc[0:,:].values.tolist()
            train, test = train_test_split(df_list, test_size=test_size, train_size=train_size)
            actual = [row[-1] for row in test]
            for row in test:
                row[-1] = None
            pred = [predict_label(train, row) for row in test]
        # Sklearn Algorithm
        else:
            train, test = train_test_split(df, test_size=test_size, train_size=train_size)
            actual = test['Species'].tolist()
            gnb = GaussianNB()
            pred = gnb.fit(train.loc[:,train.columns!='Species'].values, train['Species']).predict(test.loc[:,test.columns!='Species'])            
        actuals.extend(actual)
        preds.extend(pred)
        accuracies.append(accuracy_metric(actual, pred))
    avg = str(round(sum(accuracies)/float(len(accuracies)), 2))
    # Return average accuracy
    if return_vals:
        return avg
    # Print data analysis of the results
    else:
        print("-----------------------------------------------------------------------------------------------")
        print("Model: " + algorithm + "\tTest Size: %d%%\tTrain Size: %d%%" % (test_size * 100, train_size * 100))
        print("Average accuracy is " + avg + "%"," ran", runs, "time(s)")
        cf_matrix = confusion_matrix(actuals, preds)
        fig, ax = plt.subplots(figsize=(3,3))
        group_counts = ["{0:0.0f}".format(value) for value in
                        cf_matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                             cf_matrix.flatten()/np.sum(cf_matrix)]
        labels = [f"{v1}\n{v2}" for v1, v2 in
                  zip(group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(cf_matrix.shape[0],cf_matrix.shape[1])
        ticks = ['setosa', 'versilor', 'virginica']
        sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='binary', cbar=False, xticklabels=ticks, yticklabels=[x for x in ticks])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        print("\nClassification Report:")
        print(classification_report(actuals, preds))


# In[6]:


# Based on lists of the actual values and predicted values, evaluate how closely they match
def accuracy_metric(actual, predicted):
    matching_value_count = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            matching_value_count += 1
    return matching_value_count / float(len(actual)) * 100


# In[7]:


# Normalizes a dataframe from 0 to given range.
# Dependent feature should be listed last, and will not be normalized.
def normalize(df, range_=1):
    col = df.columns[-1]
    df_norm = df.copy()
    df_norm.pop(col)
    df_norm = ((df_norm-df_norm.min()) / (df_norm.max()-df_norm.min())) * range_
    df_norm[col] = df.iloc[:,-1]
    return df_norm


# ## Data Loading & Cleaning

# In[8]:


pred_col = 'Species'
# Load data
df = pd.read_csv('./Iris.csv')
og_shape = df.shape

# Move dependent variable to end
dep_col = df.pop(pred_col)
df.insert(len(df.columns), pred_col, dep_col)
# Fill missing values with mean of column
df.fillna(df.mean(), inplace=True)
# Drop duplicates
df.drop_duplicates(inplace=True)
# Map diagnosis string to int
diag_map = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
df = df.applymap(lambda s: diag_map.get(s) if s in diag_map else s)
# Normalize independent data
df = normalize(df, 1)

print("Number of rows removed:", og_shape[0]-df.shape[0], "\n")
print(df.info())
display(df.sample(3))


# ## Data Analysis

# In[9]:


# Visuals of feature correlation
def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_facecolor('whitesmoke')
    plt.ylim(0,20)
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    mask = mask[1:, :-1]
    sns.heatmap(corr.iloc[1:,:-1], mask=mask, vmin=-1, vmax=1, square=True, cmap="binary", annot=True, center=0)
    plt.show()
    sns.heatmap(corr[['Species']],vmin=-1, vmax=1, square=True, cmap="binary", annot=True, center=0)
    plt.show()

plot_heatmap(df)
CorrMatrixAnalysis(df, 'Species')


# There are very few features in the dataset, and 4/5 of them have high correlation with the dependent feature "Species". Since there is such a low feature count, I'm hesitant to confidently drop any of them, but I wanted to see if there's any discernable difference if the low correlation feature is dropped.

# In[15]:


print("All Features (5)")
evaluate_model(df, 50, "default", .2, .8)
print("Only High Correlation Features (4)")
evaluate_model(df.copy().loc[:,df.columns!='SepalWidthmCm'], 50, "default", .2, .8)


# I ran that test multiple times, and there was some variance, but all features was consistently more accurate than just high correlation features. I'm not sure if this is due to overfitting, but I decided to keep the low correlation feature.
# 
# "Id" is a feature that is typically dropped from datasets that I work with. However, this one has a very high correlation to Species. Going with the numbers, "Id" shouldn't be dropped, but I wanted to compare accuracies if it was. 

# In[11]:


print("All Features (5)")
evaluate_model(df, 50, "default", .2, .8)
print("All Features Except 'Id' (4)")
evaluate_model(df.copy().loc[:,df.columns!='Id'], 50, "default", .2, .8)


# This change had the most significant impact on accuracy, in a negative way. My theory is that "Id" is significant because it is a very organized, ordered dataset. 

# ## Parameter & Algorithm Testing

# In[12]:


# Evaluate average accuracy with all reasonable splits of train & test, for both algorithms.
train_sizes = [.9, .8, .7, .6, .5, .4, .3, .2, .1]
accuracies = [evaluate_model(df, runs=20, algorithm='default', test_size=round(1-size, 2), train_size=size, return_vals=True) for size in train_sizes]
sklearn_accuracies = [evaluate_model(df, runs=20, algorithm='sklearn', test_size=round(1-size, 2), train_size=size, return_vals=True) for size in train_sizes]


# In[13]:


# Visual plots of the above information gathered
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('My Algorithm', fontsize=16)
ax1.set_facecolor('whitesmoke')
ax2.set_facecolor('whitesmoke')

x = [len(df)-(len(df)*size) for size in train_sizes[::-1]]
ax1.plot(x, accuracies, c='purple')
ax1.set_xlabel('# samples in train set')
ax1.set_ylabel('accuracy %')

x = [i*100 for i in train_sizes]
ax2.plot(x, accuracies, c='purple')
ax2.set_xlabel('train size %')
ax2.set_ylabel('accuracy %')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Sklearn Algorithm', fontsize=16)
ax1.set_facecolor('whitesmoke')
ax2.set_facecolor('whitesmoke')

x = [len(df)-(len(df)*size) for size in train_sizes[::-1]]
ax1.plot(x, sklearn_accuracies, c='purple')
ax1.set_xlabel('# samples in train set')
ax1.set_ylabel('accuracy %')

x = [i*100 for i in train_sizes]
ax2.plot(x, sklearn_accuracies, c='purple')
ax2.set_xlabel('train size %')
ax2.set_ylabel('accuracy %')
plt.show()


# Both algorithms had a comperable correlation between accuracy and train size. Accuracy improved when the number of samples in the training set increased - however, this is probably due to overfitting, so a reasonable balance has to be chosen. I think 80% train, 20% test is a fair split.

# ## Final Evaluations

# In[14]:


print("My Algorithm")
evaluate_model(df, runs=20, algorithm='default', test_size=.2, train_size=.8, return_vals=False)
print("Sklearn Algorithm")
evaluate_model(df, runs=20, algorithm='sklearn', test_size=.2, train_size=.8, return_vals=False)


# Both algorithms are very accurate, with my algorithm being consistently slightly more accurate. Versicolor was the most commonly incorrectly predicted.
