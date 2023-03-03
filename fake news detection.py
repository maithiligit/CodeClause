#!/usr/bin/env python
# coding: utf-8

# ## AUTHOR - PAGARE MAITHILI
# ### DATA SCIENCE INTERN AT CODE CLAUSE
# ### TASK1 - FAKE NEWS DETECTION
# #### DATASET- https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view
# ### Import Libraries

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('news.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.duplicated().sum()


# In[7]:


df = df.drop_duplicates()


# In[8]:


df.duplicated().sum()


# In[9]:


df.isnull().sum()


# In[10]:


df=df.drop('Unnamed: 0',axis=1)


# In[11]:


df.sample(10)


# In[12]:


l = df.label;
l.head()


# In[13]:


df.describe()


# In[14]:


df=df.drop('title',axis=1)


# In[15]:


df.sample(10)


# ## DATA VISUALIZATION

# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[17]:


plt.pie(df['label'].value_counts(),labels=['Real','Fake'],colors=['red','yellow'],autopct='%0.3f')
plt.show()


# ### Data is well distributed
# 

# In[18]:


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
df['label']=le.fit_transform(df['label'])
df.sample(10)


# ### 1 means fake and 0 means real
# 

# In[22]:


x=df['text']
y=df['label']


# In[23]:



# Splitting into training and testing dataset

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df['text'],df['label'], test_size=0.2, random_state=7)


# In[24]:


x_train


# In[25]:


# To convert Text Data to vectors

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
x_train= vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test)


# In[26]:


# Using Logistic Regression

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(x_train,y_train)
LogisticRegression()
pred_lr=LR.predict(x_test)
LR.score(x_test, y_test)


# ### TESTING
# 

# In[27]:


def answer(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def test(n):
    test_n = {"text":[n]}
    new_def_test = pd.DataFrame(test_n)
    new_def_test["text"] = new_def_test["text"].apply(convert) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)

    return print("Prediction: {} ".format(answer(pred_LR[0])))


# In[28]:


n="october    at   am  pretty factual except for women in the selective service  american military is still voluntary only and hasn t been a draft since vietnam war  the comment was made by a  star general of the army about drafting women and he said it to shut up liberal yahoos"
test(n) # original fake news


# In[29]:


n='shocking  michele obama   hillary caught glamorizing date rape promoters first lady claims moral high ground while befriending rape glorifying rappers infowars com   october    comments  alex jones breaks down the complete hypocrisy of michele obama and hillary clinton attacking trump for comments he made over a decade ago while the white house is hosting and promoting rappers who boast about date raping women and selling drugs in their music   rappers who have been welcomed to the white house by the obama s include  rick ross   who promotes drugging and raping woman in his song  u o n e o    while attacking trump as a sexual predator  michelle and hillary have further mainstreamed the degradation of women through their support of so called musicians who attempt to normalize rape  newsletter sign up get the latest breaking news   specials from alex jones and the infowars crew  related articles'
test(n) # original answer fake news


# ## SAVING THE MODEL
# 

# In[30]:


import pickle


# In[31]:


name 'LR' is not defined

