import os
import sys
#os.chdir(sys.argv[1])
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow_text as text

f = open(sys.arg[1])
data = json.load(f)
labels = []
patterns = []
responses = {}
for intent in data['intents']:
    for sentence in intent['patterns']:
        labels.append(intent['tag'].lower())
        patterns.append(sentence)
        responses[intent['tag'].lower()]=intent['responses']  
Classes = sorted(list(set(labels)))

Labels=[]  
for label in Classes:
 bag = [] 
 for sentence in labels:
  T = [sentence.lower()]
  if label in T:
      bag.append(1)
  else:
      bag.append(0)
 Labels.append(bag)
Labels = np.transpose(np.array(Labels))

import nltk
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
def text_preprocessing(responses):
    lemmatizer = WordNetLemmatizer()
    S=[]
    W=[]
    for sentence in responses:
        x = re.sub('[^a-zA-Z]',' ',sentence)
        x = x.lower()
        words = word_tokenize(x)
        L=[]
        for word in words:
         x = word                  #lemmatizer.lemmatize(word)
         W.append(x)
         L.append(x)
        x = ' '.join(L)
        S.append(x)
        W = sorted(list(set(W)))
    return (S,W)

processed_text,words = text_preprocessing(patterns)

def onehot(patterns,words):  
  text_num=[]  
  for sentence in patterns:
    D = []
    for i in range(len(words)):
        D.append(0)    
        H=D
    T = sentence.lower()
    T = word_tokenize(sentence)
    for x in T:
        for j in range(len(H)):
            if words[j]==x:  
                H[j]=1
    text_num.append(H)
  text_num = np.array(text_num)
  return text_num

text_num = onehot(processed_text,words)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(text_num,Labels,test_size=0.3,random_state=5)

import tensorflow as tf
import tensorflow_text as text
from keras.models import Sequential
from keras.layers import Dense, Dropout,Embedding
from keras.optimizers  import Adam,SGD
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import keras_tuner as kt
import keras
from keras_tuner.tuners import RandomSearch
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(X_train.shape[1],)))
    for i in range(hp.Int('num_layers', 2,10)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(Y_train.shape[1], activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['AUC'])
    return model

tuner = RandomSearch(
    build_model,
    kt.Objective("val_auc", direction="max"),
    max_trials=50,
    executions_per_trial=1,
    directory=r'C:\Users\guthu\Desktop\CHATBOT\hyperparameters',
    project_name='Chatbot')

tuner.search(X_train,Y_train, epochs=300,batch_size=512, validation_data=(X_test, Y_test))

models = tuner.get_best_models(num_models=1)
best_model = models[0]
best_model.build(input_shape=(X_train.shape[1],))
best_model.summary()
best_model.save('chatbot.h5')
with open('Classes.pkl','wb') as h:
    pickle.dump(Classes,h)
with open('words.pkl','wb') as h:
    pickle.dump(words,h)
with open('responses.pkl','wb') as h:
    pickle.dump(responses,h)
