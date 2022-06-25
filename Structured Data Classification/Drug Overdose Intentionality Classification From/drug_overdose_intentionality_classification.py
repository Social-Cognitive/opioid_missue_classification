#Importing modules
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
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