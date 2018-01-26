%clear

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC

train=pd.read_csv( 'train.csv', sep=',', header=None ).values
test=pd.read_csv( 'test.csv', sep=',', header=None ).values

train_features = train[1:,1:]
train_labels = train[1:,0].reshape(train.shape[0]-1,1)
test_features = test[1:,:]

train_features = train_features.astype(float)
train_features = np.nan_to_num(train_features)
train_labels = train_labels.astype(int)
test_features = test_features.astype(float)
test_features = np.nan_to_num(test_features)

pca = PCA(n_components=2)
pca.fit(train_features)


clf = SVC()
clf.fit(train_features,train_labels)
preds = clf.predict(test_features)

#temp=pd.read_csv( 'sample_submission.csv', sep=',', header=None ).values
#op[:,1] = preds
op = np.zeros((28000,2))
op[:,0] = np.arange(1,28001,1)
op[:,1] = preds
op[:,0] = op[:,0].astype(int)
op = op.astype(int)
df = pd.DataFrame( op, columns = ['ImageId','Label'])
#op[:,0] = op[:,0].astype(int)
#op[:,1] = op[:,1].astype(int)
#df['ImageId'] = op[:,0]
#df['Label'] = op[:,1]
df.to_csv("res.csv",index=False)