import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer


## Mapping to a Uniform distribution
from sklearn.model_selection import train_test_split
data = pd.load_csv('data.csv')
X = data[data[:,:-1]]
y = data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)
np.percentile(X_train[:, 0], [0, 25, 50, 75, 100])

np.percentile(X_train_trans[:, 0], [0, 25, 50, 75, 100])
np.percentile(X_test[:, 0], [0, 25, 50, 75, 100])
np.percentile(X_test_trans[:, 0], [0, 25, 50, 75, 100])

## Mapping to a Gaussian distribution
pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
X_lognormal = np.random.RandomState(616).lognormal(size=(3, 3))
X_lognormal

quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
X_trans = quantile_transformer.fit_transform(X)
quantile_transformer.quantiles_

