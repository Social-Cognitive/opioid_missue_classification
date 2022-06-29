SEED = 123456
import os
import random as rn
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
tf.random.set_seed(SEED)

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)

rn.seed(SEED)
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10)
mc = ModelCheckpoint('model_best.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)


import pandas as pd

df = pd.read_csv("Unstructured_data_455.csv")
#df = pd.read_csv('prepro_unstructured data.csv')
print(df.shape)
df.head(5)
x = df_p.Text.values
y = df_p['Intentional'].values
x_train = x.reshape(len(x),1)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from models import *

maxlen = 2000
encode_dim = 128
batch_size = 32
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train.ravel())
tokenized_word_list = tokenizer.texts_to_sequences(x_train.ravel())
vocab_size = len(tokenizer.word_index)
X_train_padded = pad_sequences(tokenized_word_list, maxlen = maxlen, padding='post')

from sklearn.model_selection import train_test_split
X_train_final2, X_val, y_train_final2, y_val = train_test_split(X_train_padded, y, test_size = 0.2,random_state = 42)

#basic lstm
print('*'*50)
print('Training basic lstm model')
print('*'*50)

model1 = basic_lstm(vocab_size=vocab_size, X_train_padded=X_train_padded)

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model1.compile(opt, loss = "binary_crossentropy", metrics=["accuracy"])
history = model1.fit(X_train_final2, y_train_final2, epochs = 10, batch_size = batch_size, verbose = 1, validation_data = [X_val, y_val],callbacks = [es, mc])

from plotting import *
import plotting
plot_loss_and_acc(history)
performance(model1,X_train_final2, X_val, y_train_final2, y_val,'model_best.h5')


#cnn
print('*'*50)
print('Training cnn based model')
print('*'*50)
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10)
mc = ModelCheckpoint('model_best_cnn.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)
model1 = cnn(vocab_size=vocab_size)

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model1.compile(opt, loss = "binary_crossentropy", metrics=["accuracy"])
history = model1.fit(X_train_final2, y_train_final2, epochs = 10, batch_size = batch_size, verbose = 1, validation_data = [X_val, y_val],callbacks = [es, mc])

plot_loss_and_acc(history)
performance(model1,X_train_final2, X_val, y_train_final2, y_val,'model_best_cnn.h5')

#cnn
print('*'*50)
print('Training attention based model')
print('*'*50)
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10)
mc = ModelCheckpoint('model_best_attention.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)
model1 = attention_contex(X_train_padded,vocab_size)

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model1.compile(opt, loss = "binary_crossentropy", metrics=["accuracy"])
history = model1.fit(X_train_final2, y_train_final2, epochs = 10, batch_size = batch_size, verbose = 1, validation_data = [X_val, y_val],callbacks = [es, mc])

plot_loss_and_acc(history)
performance(model1,X_train_final2, X_val, y_train_final2, y_val,'model_best_attention.h5')