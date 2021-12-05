import numpy as np
from numpy import genfromtxt

from sklearn.svm import OneClassSVM

import _pickle as cPickle

# Data
num_joint = 7
num_seq = 10

#train_data = genfromtxt('./data/TrainingData.csv', delimiter=',')
train_data = genfromtxt('./data/TestingData.csv', delimiter=',')

clf = OneClassSVM(gamma=0.001, nu=0.001).fit(abs(train_data))

# save the classifier
with open('./model/ocsvm_residual.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)    

print("Training Finished!")