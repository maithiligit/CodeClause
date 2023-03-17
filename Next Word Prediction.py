#!/usr/bin/env python
# coding: utf-8

# ## AUTHOR-PAGARE MAITHILI
# ### DATA SCIENCE INTERN AT CODECLAUSE
# ### TASK 4-Next Word Prediction
# ### Importing The Required Libraries:

# In[7]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from IPython.display import Image 
import pickle
import numpy as np
import os


# In[2]:


file = open("metamorphosis_clean.txt", "r", encoding = "utf8")
lines = []

for i in file:
    lines.append(i)
    
print("The First Line: ", lines[0])
print("The Last Line: ", lines[-1])


# ### Cleaning the data:

# In[3]:


data = ""

for i in lines:
    data = ' '. join(lines)
    
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '')
data[:360]


# In[4]:


import string

translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
new_data = data.translate(translator)

new_data[:500]


# In[5]:


z = []

for i in data.split():
    if i not in z:
        z.append(i)
        
data = ' '.join(z)
data[:500]


# ### Tokenization:

# In[6]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# saving the tokenizer for predict function.
pickle.dump(tokenizer, open('tokenizer1.pkl', 'wb'))

sequence_data = tokenizer.texts_to_sequences([data])[0]
sequence_data[:10]


# In[7]:


vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)


# In[8]:


sequences = []

for i in range(1, len(sequence_data)):
    words = sequence_data[i-1:i+1]
    sequences.append(words)
    
print("The Length of sequences are: ", len(sequences))
sequences = np.array(sequences)
sequences[:10]


# In[9]:


X = []
y = []

for i in sequences:
    X.append(i[0])
    y.append(i[1])
    
X = np.array(X)
y = np.array(y)


# In[10]:


print("The Data is: ", X[:5])
print("The responses are: ", y[:5])


# In[11]:


y = to_categorical(y, num_classes=vocab_size)
y[:5]


# ### Creating the Model:

# In[12]:


model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))


# In[13]:


model.summary()


# ### Plot The Model:

# In[14]:


from tensorflow import keras
from keras.utils.vis_utils import plot_model

keras.utils.plot_model(model, to_file='model.png', show_layer_names=True)


# ### Callbacks:

# In[15]:


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard

checkpoint = ModelCheckpoint("nextword1.h5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto')

reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)

logdir='logsnextword1'
tensorboard_Visualization = TensorBoard(log_dir=logdir)


# ### Compile The Model:

# In[16]:


model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001))


# ### Fit The Model:

# In[17]:


model.fit(X, y, epochs=150, batch_size=64, callbacks=[checkpoint, reduce, tensorboard_Visualization])


# ### Graph:

# In[10]:


# https://stackoverflow.com/questions/26649716/how-to-show-pil-image-in-ipython-notebook
# tensorboard --logdir="./logsnextword1"
# http://DESKTOP-U3TSCVT:6006/

from IPython.display import Image 
pil_img = Image(filename='graph.png')
display(pil_img)


# ### Observation:
# ### We are able to develop a decent next word prediction model and are able to get a declining loss and an overall decent performance.
