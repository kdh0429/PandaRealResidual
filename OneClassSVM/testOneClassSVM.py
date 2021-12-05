import numpy as np
from numpy import genfromtxt

from sklearn.svm import OneClassSVM

import _pickle as cPickle

# Model
with open('./model/ocsvm_residual.pkl', 'rb') as fid:
    clf = cPickle.load(fid)
print("Loaded Model!")

# Data
train_data = genfromtxt('./data/TrainingData.csv', delimiter=',')
test_data = genfromtxt('./data/TestingData.csv', delimiter=',')

residual_train = abs(train_data)
residual_test = abs(test_data)

np.savetxt('./result/training_result.txt',clf.decision_function(residual_train))
np.savetxt('./result/testing_result.txt',clf.decision_function(residual_test))

user_test_data = np.abs(np.array([[1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]]))

print(clf.decision_function(user_test_data))