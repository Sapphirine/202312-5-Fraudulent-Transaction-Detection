# Libraries
import numpy as np
import pandas as pd
import os
import gc
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
# ML packages

from sklearn.svm import LinearSVC

PATH = '/home/ggvkulkarni/proj/data/'

X_train = pd.read_csv(PATH + "xtrain.csv")
y_train = pd.read_csv(PATH + "ytrain.csv")
X_test = pd.read_csv(PATH + "xtest.csv")
y_test = pd.read_csv(PATH + "ytest.csv")

# Linear SVC
svc_clf = LinearSVC(random_state=0)
#svc_clf = CalibratedClassifierCV(clf)
svc_clf.fit(X_train, y_train)

y_pred = svc_clf.predict(X_test)

from sklearn.metrics import  accuracy_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

#y_pred_proba = svc_clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
#auroc = roc_auc_score(y_test, y_pred_proba)

print(f"{accuracy:.4f},{precision:.4f},{recall:.4f},{f1:.4f}")


