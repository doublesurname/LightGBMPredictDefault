#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df= pd.read_csv("Credit Card Default II (balance).csv")


# In[3]:


#Check for null values
df.isnull().sum()


# In[4]:


#Check for negative number
df.lt(0).sum()


# In[5]:


#Replace negative age with na as it is an invalid data value
df["age"] = df.age.apply(lambda x: x if x > 0 else np.nan)


# In[6]:


#Check that is is sucessfully replace with na
df.head(16)


# In[7]:


#Check that is is sucessfully replace with na
df.isnull().sum()


# In[8]:


# Check the distribution of the target variable
df['default'].value_counts()
# Data distribution shows that half will default, half will not default


# In[9]:


X = df.loc[:,["income", "age", "loan"]]
Y = df.loc[:,["default"]]


# In[10]:


# Split the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)


# In[11]:


# Build the lightgbm model
import lightgbm as lgb
model = lgb.LGBMClassifier()
model.fit(X_train, Y_train)


# In[12]:


# Predict the results
Y_pred=model.predict(X_test)


# In[13]:


# View accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_pred, Y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(Y_test, Y_pred)))
#LightGBM Model accuracy score: 0.9922 which means the model is very highly accurate


# In[14]:


#Compare train and test set accuracy
Y_pred_train = model.predict(X_train)
print('Training set accuracy score: {0:0.4f}'. format(accuracy_score(Y_train, Y_pred_train)))

#The training and test set accuracy are quite comparable. So, we cannot say there is overfitting.


# In[15]:


# View confusion-matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives = ', cm[0,0])
print('\nTrue Negatives = ', cm[1,1])
print('\nFalse Positives = ', cm[0,1])
print('\nFalse Negatives = ', cm[1,0])

#Output shows that model is highly accurate


# In[16]:


# Visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[17]:


#Classification metrics
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

#Model has 99% accuracy


# In[18]:


#The three parameters are num_leaves, min_data_in_leaf and max_depth. They are set at default values 31, 20,-1 respectively.
#Since the model is build using LightGBM, removing outlier, normalization and removing NA, non-number and missing data is not required.


# In[24]:


import joblib 


# In[25]:


joblib.dump(model, "creditcarddefault2")


# In[ ]:




