%clear

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

train=pd.read_csv( 'train.csv', sep=',', header=None ).values
test=pd.read_csv( 'test.csv', sep=',', header=None ).values

train_features = train[1:,[0,1,3,4,17,18,19,20,26,34,36,37,38,43,44,45,46,47,48,49,50,51,52,54,56,59,61,62,66,67,68,69,70,71,75,76,77]]
train_labels = train[1:,80]#.reshape(train.shape[0]-1,1)
test_features = test[1:,[0,1,3,4,17,18,19,20,26,34,36,37,38,43,44,45,46,47,48,49,50,51,52,54,56,59,61,62,66,67,68,69,70,71,75,76,77]]

train_features = train_features.astype(float)
train_features = np.nan_to_num(train_features)
train_labels = train_labels.astype(float)
test_features = test_features.astype(float)
test_features = np.nan_to_num(test_features)

reg = GradientBoostingRegressor()
reg.fit(train_features,train_labels)
preds = reg.predict(test_features)

op = np.zeros((1459,2))
op[:,0] = test[1:,0]
op[:,1] = preds
op[:,0] = op[:,0].astype(int)
df = pd.DataFrame( op, columns = ['Id','SalePrice'])
df['Id'] = test[1:,0]
df.to_csv("res.csv",index=False)
