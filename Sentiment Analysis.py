#!/usr/bin/env python
# coding: utf-8

# ## AUTHOR -  PAGARE MAITHILI
# ### DATA SCIENCE INTERN AT CODE CLAUSE
# ### TASK 2 -SENTIMENT ANALYSIS
# #### DATASET- https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset
# ### Import Libraries

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# Importing the csv

df = pd.read_csv('test.csv', header= 0, encoding= 'unicode_escape')
df.text=df.text.astype(str)


# In[3]:


# First 5 rows of the dataset

df.head()


# In[4]:


# Last 5 rows of the dataset

df.tail()


# In[5]:


# Checking if there are any null values

df.isnull().sum()


# In[6]:


# Information of the dataset

df.info()


# In[7]:


# Shape of the dataset

df.shape


# In[8]:


# Chceking for duplicate values

df.duplicated().sum()


# In[9]:


df = df.drop_duplicates()
df.duplicated().sum()


# In[10]:


df.shape


# In[11]:


# Eliminationg the unnecessary columns

df.drop(['textID', 'Country','Population -2020','Land Area (Km²)','Density (P/Km²)'], axis=1, inplace=True)


# In[12]:


# New dataframe

df.sample(10)


# In[13]:


# Checking how many outputes do we have

df['sentiment'].unique()


# In[14]:


df.isnull().sum()


# ### DATA VISUALIZATION
# 

# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[16]:


sns.histplot(df['sentiment'],kde=True)


# In[17]:


#pie chart

plt.pie(df['sentiment'].value_counts(),labels=['Neutral','Positive','Negative'],autopct='%0.3f')
plt.show()


# ### Data is well distributed
# 

# In[18]:


# function to change the texts (title,text) machine understandable

import re

def convert(text):
    text = text.lower()
    text = re.sub(r'https?://S+|www\.\S+' , '' , text)
    text = re.sub('\n' , '' , text)
    text = re.sub('\[.*?\]', '', text)
    words = []
    for i in text:
        if i not in string.punctuation:
            words.append(i)
    return ''.join(words)


# In[19]:


import string

df["text"] = df["text"].apply(convert)


# In[20]:


df


# ### Making Data Ready for Model fitting
# 

# In[21]:


#to make the label neumerical from categorical

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['sentiment']=le.fit_transform(df['sentiment'])
df.head()


# In[22]:


df.drop(['Time of Tweet', 'Age of User'], axis=1, inplace=True)


# In[23]:


df.head()


# In[24]:


new = df.to_csv('new.csv')


# In[25]:


x=df['text']
y=df['sentiment']


# In[26]:


# Making the Traing and testing dataset

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x, y, test_size=0.33, random_state=42)


# In[27]:


# To convert Text Data to bag of words

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
x_train= vectorizer.fit_transform(X_train)
x_test = vectorizer.transform(X_test)


# In[28]:


# To convert Text Data to bag of words

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
x_train= vectorizer.fit_transform(X_train)
x_test = vectorizer.transform(X_test)


# In[29]:


#  Multinomial naive Bayes

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(x_train , y_train)
model=nb.fit(x_train,y_train)
prediction=model.predict(x_test)
accuracy_score(y_test,prediction)


# In[30]:


def answer(n):
    if n == 0:
        return "Negative"
    elif n == 1:
        return "Neutral" 
    elif n == 2:
        return "Positive"
    
def testing(text):
    testing_text = {"text":[text]}
    new_def_test = pd.DataFrame(testing_text)
    new_def_test["text"] = new_def_test["text"].apply(convert) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)
    prediction = model.predict(new_xv_test)

    return print("Prediction: {} ".format(answer(prediction[0])))


# In[31]:


text='happy bday!'
testing(text) # original answer positive


# In[32]:


text='and within a short time of the last clue all of them'
testing(text) # original answer neutral


# In[33]:


text='im in va for the weekend my youngest son turns 2 tomorrowit makes me kinda sad he is getting so big check out my twipics'
testing(text) # original answer negative


# ### Saving The Model
# 

# In[34]:


import pickle
pickle.dump(model,open('sentiment.pkl','wb'))

