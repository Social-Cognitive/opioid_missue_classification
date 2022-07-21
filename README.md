
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


## Unstructured data classification

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

p_values.sort_values(ascending=True)
```
![](Structured%20Data%20Classification/Images/p-values.png)
```python
f_values.sort_values(ascending=False)
```
![](Structured%20Data%20Classification/Images/f-values.png)

### Dataset spliting into training and testing
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

"""#AdaBoostClassifier Implementation"""

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

#Clearing the lists so that we can keep performance values for the next classifier
train_accuracy.clear()
test_accuracy.clear()

train_mcc.clear()
test_mcc.clear()

train_sensitivity.clear()
test_sensitivity.clear()

train_specificity.clear()
test_specificity.clear()

"""#LogisticRegressionClassifier"""

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

#Clearing the lists so that we can keep performance values for the next classifier
train_accuracy.clear()
test_accuracy.clear()

train_mcc.clear()
test_mcc.clear()

train_sensitivity.clear()
test_sensitivity.clear()

train_specificity.clear()
test_specificity.clear()

"""#SupportVectorClassifier"""

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

#Clearing the lists so that we can keep performance values for the next classifier
train_accuracy.clear()
test_accuracy.clear()

train_mcc.clear()
test_mcc.clear()

train_sensitivity.clear()
test_sensitivity.clear()

train_specificity.clear()
test_specificity.clear()

"""#XGB Classifier"""

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

#Clearing the lists so that we can keep performance values for the next classifier
train_accuracy.clear()
test_accuracy.clear()

train_mcc.clear()
test_mcc.clear()

train_sensitivity.clear()
test_sensitivity.clear()

train_specificity.clear()
test_specificity.clear()

"""#Random Forest Classifier"""

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
