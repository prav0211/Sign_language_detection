import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

#Generating the confusion matrix!
y_test = pd.read_csv("C:\Users\prave\Documents\Mini project 1 sign")
y_pred = pd.read_csv("C:\Users\prave\Documents\Mini project 1 sign")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))