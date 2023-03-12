#!/usr/bin/env python
# coding: utf-8

# ## AUTHOR - PAGARE MAITHILI
# ### DATA SCIENCE INTERN AT CODECLAUSE
# ### TASK 3 - DETECTION OF THE PARKINSON'S DISEASE
# #### DATASET- https://www.kaggle.com/datasets/dipayanbiswas/parkinsons-disease-speech-signal-features
# ### Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import xgboost as xgb
import os, sys
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


#DataFlair - Read the data
df=pd.read_csv("pd_speech_features.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


for col in df.columns:
    if df[col].dtypes == 'object':
        print(col, df[col].unique())


# In[7]:


df.isna().sum().sum()


# In[8]:


df['class'].value_counts()


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))
sns.heatmap(df.corr())


# In[10]:


def preprocess_inputs(df):
    df = df.copy()
    df = df.drop('id',axis=1)
    
    y = df['class']
    X = df.drop('class',axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, shuffle=True, random_state=43)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns, index = X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns, index = X_test.index)
    
    return X_train, X_test, y_train, y_test


# In[11]:


X_train,X_test,y_train,y_test = preprocess_inputs(df)
X_train


# In[12]:


from imblearn.over_sampling import RandomOverSampler, SMOTE

oversampler = SMOTE(random_state=1)
X_train_smote, y_train_smote = oversampler.fit_resample(X_train, y_train)


# In[13]:


models = {
    '           Linear SVM': LinearSVC(),
    '        XGBClassifier': xgb.XGBClassifier(),
    '    Gradient Boosting': GradientBoostingClassifier(),
    '        Decision Tree': DecisionTreeClassifier(),
    '        Random Forest': RandomForestClassifier(),
    ' KNeighborsClassifier': KNeighborsClassifier(),
    '   Bagging Classifier': BaggingClassifier()
}

for name, model in models.items():
    model = model.fit(X_train_smote, y_train_smote)
    print(name + " trained")


# In[14]:


for name, model in models.items():
    print(name + ": {:.2f}%".format(model.score(X_test, y_test) * 100))

