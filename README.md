
# Machine Learning based Approach to Classify Opioid Patients from the Structured and Unstructured Dataset

 The opioid crisis shows that in recent years the number of drug overdoses has increased in the United States. As a result, the US has been experiencing a large number of deaths associated with drug misuse. In this study, we will be shocasing the classification of intentional and unintentional opioid users for both structured(tabular) and unstructured datasets. The classification models have been trained over 454 patient instances.

## Installing software and files
To do the project, we need to install some softwares and files. In this regard, we will be doing all the implementations in Python language on jupyter notebook. To install jupyter notebook and launch other application and files at first we have to download Anaconda which is free.

Link to Download Anaconda : https://www.anaconda.com/?modal=nucleus-commercial

Guideline for installing Anaconda : https://www.geeksforgeeks.org/how-to-install-anaconda-on-windows/

Once Anaconda is downloaded and installed successfully, we may proceed to download Jupyter notebook.

## Download and Install Jupyter Notebook
Link to download Jupyter using Anaconda : https://docs.anaconda.com/ae-notebooks/4.3.1/user-guide/basic-tasks/apps/jupyter/

More informations : https://mas-dse.github.io/startup/anaconda-windows-install/

Guideline to use Jupyter notebook : https://www.dataquest.io/blog/jupyter-notebook-tutorial/

## Using Google Colaboratory
For implementing the project with no configuration we can use Google Colaboratory as well.

## Installing Python libraries and packages
The required python libraries and packages are,
- pandas
- Numpy
- sklearn
- KNNImputer
- confusion_matrix
- matthews_corrcoef
- AdaBoostClassifier
- RandomForestClassifier
- SVC
- LogisticRegression
- XGBClassifier
- XGBClassifier
- accuracy_score
- tensorflow
- keras
- Perceptron
- RidgeClassifier
- SGDClassifier
- KNeighborsClassifier
- BernoulliNB


## Structured data(tabular data) classification

### Label encoding
Label encoding has been done to replace the textual data with numeric values 
```Python
import pandas as pd
import numpy as np

data = pd.read_csv("Drug_overdose_data.csv", encoding_errors='ignore')
# Preview the first 5 lines of the loaded data 
data.head()
```
![](Structured%20Data%20Classification/Images/raw%20data%20head.png)
```python
npy = data.to_numpy()
data = data[['High level category(icd-9)','Age Category','gender','discharge_location','insurance','marital_status','ethnicity','High level category(diagnosis)','los_category','Mental_status','Intentional']]


#Label encoding (Replacing strings with numeric values)
ordinal_label = {k: i for i, k in enumerate(data['High level category(icd-9)'].unique(), 0)}
ordinal_label
```
![](Structured%20Data%20Classification/Images/label%20encoding%20for%20High%20level%20category(icd-9).png)
```python
ordinal_label = {'Diseases of blood and circulatory system': 7,
 'Diseases of digestive system': 4,
 'Diseases of genitourinary system': 1,
 'Diseases of nervous system and Mental Disorder': 3,
 'Diseases of respiratory system': 5,
 'Endocrine, Metabolic, Immunity Disorder and Sepsis': 6,
 'Poisoning and Injury': 0,
 'Skin, Subcutaneous tissue and Musculoskeletal diseases ': 2}

data['High level category(icd-9)'] = data['High level category(icd-9)'].map(ordinal_label)

#This process will be continued for the rest of the attributes

#Saving the label encoded file
data.to_csv('pre-processed_with_Nan.csv')
```

### Impute missing values by using KNNImputer
Replacing missing values with the closest instances' values
```python
import pandas as pd
import numpy as np

data = pd.read_csv("pre-processed_with_Nan.csv", encoding_errors='ignore')
# Preview the first 5 lines of the loaded data 
data.head()
```
![](Structured%20Data%20Classification/Images/label%20encoded%20data%20head.png)
```python
from sklearn.impute import KNNImputer

#Imputing each of the missing values with the closest neighbors' values
imputer = KNNImputer(n_neighbors=1)
df = imputer.fit_transform(data)

np.save("Drug_overdose_data.npy",df)#Storing the processed data
```

### Feature selection(p-value and f-value findings)
Finding out the p-value and f-value for selecting the features to train the model
```Python
import pandas as pd
import numpy as np

data = pd.read_csv("pre-processed_with_Nan.csv", encoding_errors='ignore')
# Preview the first 5 lines of the loaded data 
data.head()

data = data[['High level category(icd-9)','Age Category','gender','discharge_location','insurance','marital_status','ethnicity','High level category(diagnosis)','los_category','Mental_status','Intentional']]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data[['High level category(icd-9)','Age Category','gender','discharge_location','insurance','marital_status','ethnicity','High level category(diagnosis)','los_category','Mental_status']], data['Intentional'],test_size=0.2,random_state=100)


from sklearn.feature_selection import chi2
f_p_values = chi2(x_train, y_train)

p_values=pd.Series(f_p_values[1])
p_values.index = x_train.columns

f_values=pd.Series(f_p_values[0])
f_values.index = x_train.columns
```
```python
p_values.sort_values(ascending=True)
```
![](Structured%20Data%20Classification/Images/p-values.png)
```python
f_values.sort_values(ascending=False)
```
![](Structured%20Data%20Classification/Images/f-values.png)

### Dataset spliting into training and testing (80:20)
Spliting the whole dataset into training and testing datasets
```python
import pandas as pd
import numpy as np

data = np.load('Drug_overdose_data.npy')

np.random.shuffle(data)
train, test = data[:int(len(data)*0.8),:], data[int(len(data)*0.8):,:]

x_train = train[:,:-1]
y_train = train[:,-1:]

x_test = test[:,:-1]
y_test = test[:,-1:]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

np.save("x_train.npy",x_train)
np.save("y_train.npy",y_train)
np.save("x_test.npy",x_test)
np.save("y_test.npy",y_test)
```

### Implementing tabular data classification (Training and Testing)
Here, we have implemented several machine learning classification algorithms to build the model and classify the data

```python
#Importing modules
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#Loading training and testing data
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

#Considering attributes with a p-value of less than 0.161
x_train = x_train[:,-4:]
x_train = np.delete(x_train, 2, 1)
x_test = x_test[:,-4:]
x_test = np.delete(x_test, 2, 1)

#defining classifiers
adaboost_classifier = AdaBoostClassifier()
logistic_regression = LogisticRegression(max_iter=500) #default a error dicche
xgb_classifier = XGBClassifier()
random_forest_classifier = RandomForestClassifier()
svc = SVC()

#Definiing few lists for storing performance
train_accuracy = []
test_accuracy = []

train_mcc = [] #The Matthews correlation coefficient (MCC)
test_mcc = []

train_sensitivity = []
test_sensitivity = []

train_specificity = []
test_specificity = []
```
### AdaBoostClassifier Implementation
```python
###############AdaBoostClassifier#################
adaboost_classifier.fit(x_train, y_train)#machine k train korlam
train_pred = adaboost_classifier.predict(x_train)
test_pred = adaboost_classifier.predict(x_test)


print(type(train_pred), type(test_pred))
print(train_pred.shape, test_pred.shape)

        
        
###############ACCURACY#################
train_accuracy.append(accuracy_score(y_train, train_pred))
test_accuracy.append(accuracy_score(y_test, test_pred))


###############MCC#################
train_mcc.append(matthews_corrcoef(y_train, train_pred))
test_mcc.append(matthews_corrcoef(y_test, test_pred))


###############Sensitivity & Specificity#################
tn, fp, fn, tp = confusion_matrix(y_train, train_pred).ravel()
train_sensitivity.append(tn/(tn+fp))
train_specificity.append(tp/(tp+fn))

tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
test_sensitivity.append(tn/(tn+fp))
test_specificity.append(tp/(tp+fn))

print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)
print("Training MCC:", train_mcc)
print("Testing MCC:", test_mcc)
print("Training sensitivity:", train_sensitivity)
print("Testing sensitivity:", test_sensitivity)
print("Training specificity:", train_specificity)
print("Testing specificity:", test_specificity)
```
![](Structured%20Data%20Classification/Images/AdaBoost%20Performance.png)
```python
#Clearing the lists so that we can keep performance values for the next classifier
train_accuracy.clear()
test_accuracy.clear()

train_mcc.clear()
test_mcc.clear()

train_sensitivity.clear()
test_sensitivity.clear()

train_specificity.clear()
test_specificity.clear()

```
### LogisticRegressionClassifier Implementation
```python

###############LogisticRegression#################
logistic_regression.fit(x_train, y_train)#machine k train korlam
train_pred = logistic_regression.predict(x_train)
test_pred = logistic_regression.predict(x_test)


print(type(train_pred), type(test_pred))
print(train_pred.shape, test_pred.shape)

        
        
###############ACCURACY#################
train_accuracy.append(accuracy_score(y_train, train_pred))
test_accuracy.append(accuracy_score(y_test, test_pred))


###############MCC#################
train_mcc.append(matthews_corrcoef(y_train, train_pred))
test_mcc.append(matthews_corrcoef(y_test, test_pred))


###############Sensitivity & Specificity#################
tn, fp, fn, tp = confusion_matrix(y_train, train_pred).ravel()
train_sensitivity.append(tn/(tn+fp))
train_specificity.append(tp/(tp+fn))

tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
test_sensitivity.append(tn/(tn+fp))
test_specificity.append(tp/(tp+fn))

print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)
print("Training MCC:", train_mcc)
print("Testing MCC:", test_mcc)
print("Training sensitivity:", train_sensitivity)
print("Testing sensitivity:", test_sensitivity)
print("Training specificity:", train_specificity)
print("Testing specificity:", test_specificity)
```
![](Structured%20Data%20Classification/Images/LogisticRegression%20performance.png)
```python
#Clearing the lists so that we can keep performance values for the next classifier
train_accuracy.clear()
test_accuracy.clear()

train_mcc.clear()
test_mcc.clear()

train_sensitivity.clear()
test_sensitivity.clear()

train_specificity.clear()
test_specificity.clear()

```
### SupportVectorClassifier Implementation
```python

###############SupportVectorClassifier#################
svc.fit(x_train, y_train)#machine k train korlam
train_pred = svc.predict(x_train)
test_pred = svc.predict(x_test)


print(type(train_pred), type(test_pred))
print(train_pred.shape, test_pred.shape)

        
        
###############ACCURACY#################
train_accuracy.append(accuracy_score(y_train, train_pred))
test_accuracy.append(accuracy_score(y_test, test_pred))


###############MCC#################
train_mcc.append(matthews_corrcoef(y_train, train_pred))
test_mcc.append(matthews_corrcoef(y_test, test_pred))


###############Sensitivity & Specificity#################
tn, fp, fn, tp = confusion_matrix(y_train, train_pred).ravel()
train_sensitivity.append(tn/(tn+fp))
train_specificity.append(tp/(tp+fn))

tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
test_sensitivity.append(tn/(tn+fp))
test_specificity.append(tp/(tp+fn))

print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)
print("Training MCC:", train_mcc)
print("Testing MCC:", test_mcc)
print("Training sensitivity:", train_sensitivity)
print("Testing sensitivity:", test_sensitivity)
print("Training specificity:", train_specificity)
print("Testing specificity:", test_specificity)
```
![](Structured%20Data%20Classification/Images/Support%20Vector%20Classifier%20performance.png)
```python
#Clearing the lists so that we can keep performance values for the next classifier
train_accuracy.clear()
test_accuracy.clear()

train_mcc.clear()
test_mcc.clear()

train_sensitivity.clear()
test_sensitivity.clear()

train_specificity.clear()
test_specificity.clear()

```
### XGB Classifier Implementation
```python

###############xgb_classifier#################
xgb_classifier.fit(x_train, y_train)#machine k train korlam
train_pred = xgb_classifier.predict(x_train)
test_pred = xgb_classifier.predict(x_test)


print(type(train_pred), type(test_pred))
print(train_pred.shape, test_pred.shape)

        
        
###############ACCURACY#################
train_accuracy.append(accuracy_score(y_train, train_pred))
test_accuracy.append(accuracy_score(y_test, test_pred))


###############MCC#################
train_mcc.append(matthews_corrcoef(y_train, train_pred))
test_mcc.append(matthews_corrcoef(y_test, test_pred))


###############Sensitivity & Specificity#################
tn, fp, fn, tp = confusion_matrix(y_train, train_pred).ravel()
train_sensitivity.append(tn/(tn+fp))
train_specificity.append(tp/(tp+fn))

tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
test_sensitivity.append(tn/(tn+fp))
test_specificity.append(tp/(tp+fn))

print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)
print("Training MCC:", train_mcc)
print("Testing MCC:", test_mcc)
print("Training sensitivity:", train_sensitivity)
print("Testing sensitivity:", test_sensitivity)
print("Training specificity:", train_specificity)
print("Testing specificity:", test_specificity)
```
![](Structured%20Data%20Classification/Images/XGB%20performance.png)
```python
#Clearing the lists so that we can keep performance values for the next classifier
train_accuracy.clear()
test_accuracy.clear()

train_mcc.clear()
test_mcc.clear()

train_sensitivity.clear()
test_sensitivity.clear()

train_specificity.clear()
test_specificity.clear()

```
### Random Forest Classifier Implementation
```python

###############random_forest_classifier#################
random_forest_classifier.fit(x_train, y_train)#machine k train korlam
train_pred = random_forest_classifier.predict(x_train)
test_pred = random_forest_classifier.predict(x_test)


print(type(train_pred), type(test_pred))
print(train_pred.shape, test_pred.shape)

        
        
###############ACCURACY#################
train_accuracy.append(accuracy_score(y_train, train_pred))
test_accuracy.append(accuracy_score(y_test, test_pred))


###############MCC#################
train_mcc.append(matthews_corrcoef(y_train, train_pred))
test_mcc.append(matthews_corrcoef(y_test, test_pred))


###############Sensitivity & Specificity#################
tn, fp, fn, tp = confusion_matrix(y_train, train_pred).ravel()
train_sensitivity.append(tn/(tn+fp))
train_specificity.append(tp/(tp+fn))

tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
test_sensitivity.append(tn/(tn+fp))
test_specificity.append(tp/(tp+fn))

print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)
print("Training MCC:", train_mcc)
print("Testing MCC:", test_mcc)
print("Training sensitivity:", train_sensitivity)
print("Testing sensitivity:", test_sensitivity)
print("Training specificity:", train_specificity)
print("Testing specificity:", test_specificity)
```
![](Structured%20Data%20Classification/Images/Random%20Forest%20Classifier%20performance.png)
```python
#Clearing the lists so that we can keep performance values for the next classifier
train_accuracy.clear()
test_accuracy.clear()

train_mcc.clear()
test_mcc.clear()

train_sensitivity.clear()
test_sensitivity.clear()

train_specificity.clear()
test_specificity.clear()
```



## Unstructured data classification

###Classification models
```python
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
```

```python
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

```
### LSTM based model implementation
```python

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

```
### 1D CNN based model implementation
```python
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

```
### Attention based model implementation
```python
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
```

### Traditional classifier based model implementation
```python
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
```
