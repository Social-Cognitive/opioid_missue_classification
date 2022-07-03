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
# Input for variable-length sequences of integers
import keras
import tensorflow
from keras import layers 
from attention import Attention
import keras
from keras import Model

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()

def basic_lstm(vocab_size,X_train_padded):
  inputs1=Input(shape=(X_train_padded.shape[1],))
  x1=Embedding(input_dim=vocab_size+1,output_dim=128)(inputs1)
  x1=LSTM(400,dropout=0.3,recurrent_dropout=0.2)(x1)
  outputs1=Dense(1,activation='sigmoid')(x1)
  model1=Model(inputs1,outputs1)
  return model1


def cnn(vocab_size,max_tokens = 2000):
  embed_len = 128
  #max_tokens = 800
  inputs = Input(shape=(max_tokens, ))
  embeddings_layer = Embedding(input_dim=vocab_size+1, output_dim=embed_len,  input_length=max_tokens)
  conv = Conv1D(32, 7, padding="same") ## Channels last
  dense = Dense(1, activation="sigmoid")

  x = embeddings_layer(inputs)
  x = conv(x)
  x = tensorflow.reduce_max(x, axis=1)
  output = dense(x)

  model = Model(inputs=inputs, outputs=output)
  return model

def attention_contex(X_train_padded,vocab_size):

    inputs=Input((X_train_padded.shape[1],))
    x=Embedding(input_dim=vocab_size+1,output_dim=128)(inputs)
    #x = Conv1D(32, 7, padding="same")(x) ## Channels last

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    #x = layers.Bidirectional(layers.LSTM(64))(x)
    att_out=Attention()(x)#attention()(x)

    #x = layers.Bidirectional(layers.LSTM(64))(att_out)
    #outputs=Dense(100,activation='relu',trainable=True)(att_out)
    outputs=Dense(1,activation='sigmoid',trainable=True)(att_out)
    model=Model(inputs,outputs)
    return model