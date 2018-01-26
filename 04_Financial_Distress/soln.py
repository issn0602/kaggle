%clear

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

fd=pd.read_csv('Financial_Distress.csv', sep=',', header=None ).values

features = fd[1:,3:]
labels = fd[1:,2].reshape(fd.shape[0]-1,1)
preds = np.zeros((labels.shape))

kf = KFold(n_splits=8)
reg = SGDRegressor()

for train_index, test_index in kf.split(features):
    features_train, features_test = features[train_index,:], features[test_index,:]
    labels_train, labels_test = labels[train_index,:], labels[test_index,:]
    reg.fit(features_train,labels_train)
    preds[test_index,:] = reg.predict(features_test).reshape(features_test.shape[0],1)

print (mean_squared_error(labels,preds))