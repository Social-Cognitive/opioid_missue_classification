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
def tfidftoken(data,Y,mx):
  # Importing library

  tfidf_vectorizer = TfidfVectorizer(max_df=.96, min_df=2, max_features=mx)
  # TF-IDF feature matrix - For columns "combine_df['tweet_stemmed']"
  tfidf_stem = tfidf_vectorizer.fit_transform(data)
  # prepare train and val sets first
  
  x_train,x_test,y_train,y_test=train_test_split(tfidf_stem,Y,test_size=0.20,random_state=42)
  
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
        clf.fit(x_train, y_train)
        name = clf.__class__.__name__

        print("="*30)
        print(name)

        print('****Results****')
        train_predictions = clf.predict(x_test)
        acc = accuracy_score(y_test, train_predictions)
        print("Accuracy: {:.4%}".format(acc))

        train_predictions = clf.predict_proba(x_test)
        second = [a[1] for a in train_predictions]
        ll = roc_auc_score(y_test, second)
        print("ROC_AUC: {}".format(ll))

        log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
        log = log.append(log_entry)




