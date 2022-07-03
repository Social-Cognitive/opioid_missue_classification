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


import pandas as pd
from processing import *
df = pd.read_csv("Unstructured_data_455.csv")
#df = pd.read_csv('prepro_unstructured data.csv')
print(df.shape)
df.head(5)

df_p = process(df)

process = 'all' #PROBLEM,TEST,TREATMENT

from mlstanza import *
from ml_train import *
val = []

for m in df_p['Text']:
  val.append(prepro(m,process))
df_p['clean'] = val


df_p['clean2'] = df_p['clean'].apply(lambda x: ' '.join(x))

#wordcloud
dfk = df_p[df_p.Intentional == 0]
wordcloud(dfk['clean2'],0)
#wordcloud
dfk = df_p[df_p.Intentional == 1]
wordcloud(dfk['clean2'],1)
tfidftoken(df_p['clean2'],df_p['Intentional'],800)