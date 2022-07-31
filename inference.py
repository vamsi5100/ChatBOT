import pickle 
import tensorflow as tf
import tensorflow_text as text
from keras.models import Sequential
from keras.layers import Dense, Dropout,Embedding
from keras.optimizers  import Adam,SGD
from tensorflow.keras.models import load_model
import pandas as pd
import json
import numpy as np

classes_file = open('Classes.pkl', 'rb')
words_file = open('words.pkl', 'rb')
res_file = open('responses.pkl','rb')
Classes = pickle.load(classes_file)
words = pickle.load(words_file)
model = load_model('chatbot.h5',compile = False)
responses = pickle.load(res_file)
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

from numpy.linalg import norm
def cosine_similarity(A,B):
    x = norm(A, axis=1).reshape(-1,1)
    cosine = np.dot(A,B)/(x*norm(B))
    return cosine

def model_predict(X,model,Classes):  
    p = model.predict(X)
    C=[]
    for i in range(len(p)):
       index = np.argmax(p[i])
       c = Classes[index]
       C.append(c)
    return C
count=0
while count<5:
   patterns = [str(input())]
   processed_text,_ = text_preprocessing(patterns)
   text_num = onehot(processed_text,words)
   prediction = model_predict(text_num,model,Classes)
   Res = responses[prediction[0]]
   processed_res,_= text_preprocessing(Res)
   res_num = onehot(processed_res,words)
   similarity_matrix = cosine_similarity(text_num,np.transpose(res_num))
   final_response = Res[np.argmax(similarity_matrix)]
   count+=1
   print(final_response)