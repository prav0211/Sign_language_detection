import joblib
import pandas as pd
import numpy as np

# Loading the model from the pickle file
clf = joblib.load('svm(asl)trial.pkl')
fname = pd.read_csv("C:\\Users\prave\\Documents\\test_ip.csv")
#predicting
op = clf.predict(fname)
y_test = pd.read_csv("C:\\Users\\prave\\Documents\\expected_op.csv")
np.savetxt("C:\\Users\\prave\\Documents\\predicted_op.csv", op, delimiter=",")
