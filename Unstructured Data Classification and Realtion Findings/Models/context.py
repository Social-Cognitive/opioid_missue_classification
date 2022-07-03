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
from processing import *

df = pd.read_csv("Unstructured_data_455.csv")
#df = pd.read_csv('prepro_unstructured data.csv')
print(df.shape)
df.head(5)
df_p = process(df)
x = df_p.Text.values
y = df_p['Intentional'].values
x_train = x.reshape(len(x),1)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from models import *

maxlen = 1000
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
#history = model1.fit(X_train_final2, y_train_final2, epochs = 10, batch_size = batch_size, verbose = 1, validation_data = [X_val, y_val],callbacks = [es, mc])

from plotting import *
import plotting
#plot_loss_and_acc(history)
#performance(model1,X_train_final2, X_val, y_train_final2, y_val,'model_best.h5')


#cnn
print('*'*50)
print('Training cnn based model')
print('*'*50)
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10)
mc = ModelCheckpoint('model_best_cnn.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)
model1 = cnn(vocab_size=vocab_size,max_tokens = maxlen)

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

#plot_loss_and_acc(history)
#performance(model1,X_train_final2, X_val, y_train_final2, y_val,'model_best_attention.h5')

# models
# models
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss,roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,classification_report,accuracy_score,confusion_matrix



from sklearn.feature_extraction.text import TfidfVectorizer

classifiers = [
        KNeighborsClassifier(3),
        #SVC(kernel="rbf", C=0.025, probability=True),
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100, random_state=0),
        BernoulliNB(),
        AdaBoostClassifier(),
        GradientBoostingClassifier()
        
        
        
        ]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
      clf.fit(X_train_final2, y_train_final2)
      name = clf.__class__.__name__

      print("="*30)
      print(name)

      print('****Results****')
      train_predictions = clf.predict(X_val)
      acc = accuracy_score(y_val, train_predictions)
      print("Accuracy: {:.4%}".format(acc))

      train_predictions = clf.predict_proba(X_val)
      second = [a[1] for a in train_predictions]
      ll = roc_auc_score(y_val, second)
      print("ROC_AUC: {}".format(ll))

      log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
      log = log.append(log_entry)

