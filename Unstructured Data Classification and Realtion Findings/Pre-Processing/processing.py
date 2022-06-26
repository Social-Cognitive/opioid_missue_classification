import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.pooling import GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.layers import *
from keras import backend as K
from sklearn.metrics import f1_score, confusion_matrix
from keras import initializers,regularizers,constraints

def preprocess1(x):
    y = x
    y=re.sub('\\[(.*?)\\]','',y) #remove de-identified brackets
    y=re.sub('[0-9]+\.','',y) #remove 1.2. since the segmenter segments based on this
    y=re.sub('dr\.','doctor',y)
    #y = re.sub("(\s\d+)","",y) 
    y=re.sub('m\.d\.','md',y)
    y=re.sub('admission date:','',y)
    y=re.sub('discharge date:','',y)
    y=re.sub('--|__|==','',y)
    return y

def preprocessing(df_less_n): 
    df_less_n['Text']=df_less_n['Text'].fillna(' ')
    df_less_n['Text']=df_less_n['Text'].str.replace('\n',' ')
    df_less_n['Text']=df_less_n['Text'].str.replace('\r',' ')
    df_less_n['Text']=df_less_n['Text'].apply(str.strip)
    df_less_n['Text']=df_less_n['Text'].str.lower()

    df_less_n['Text']=df_less_n['Text'].apply(lambda x: preprocess1(x))
    return df_less_n

def process(df):

  df = preprocessing(df)
  return df

import stanza
def clean_text(X):
    stop_words = set(stopwords.words("english")) 

    lemmatizer = WordNetLemmatizer()
    nlp = stanza.Pipeline(lang='en', processors={'ner':'i2b2','lemma':'mimic'},package = 'mimic')
    processed = []
    for text in X:
        text = text[0]
        text = re.sub(r'[^\w\s]','',text, re.UNICODE)
        text = re.sub('<.*?>', '', text)
        
        #print(text)
        #text = text.strip().split()
        #text = [word for word in text if not word in stop_words]
        #text = " ".join(text)
        
        #doc = nlp(text)
        #text = [word.lemma for sent in doc.sentences for word in sent.words]
        #text = " ".join(text)
        #print(text)
        processed.append(text)
    
    return processed