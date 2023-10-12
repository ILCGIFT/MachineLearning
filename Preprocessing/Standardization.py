# StandardScaler
from sklearn import preprocessing
import numpy as np
X_train = np.array([[a,b,c,d],
                    [e,f,g,h],
                    [i,j,k,l]])

scaler = preprocessing.StandardScaler().fit(X_train)
scaler

scaler.mean_
scaler.scale_

X_scaled = scaler.transform(X_train)
X_scaled
X_scaled.mean(axis=0)
X_scaled.std(axis=0)

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)  # apply scaling on training data
pipe.score(X_test, y_test)

## Scaling Feature To Range
X_train = np.array([[a,b,c,d],
                    [e,f,g,h],
                    [i,j,k,l]])
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_train_minmax
X_test_minmax = min_max_scaler.transform(X_test)
X_test_minmax
min_max_scaler.scale_
min_max_scaler.min_
max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
X_train_maxabs
X_test_maxabs = max_abs_scaler.transform(X_test)
max_abs_scaler.scale_

## Scaling sparse data
## Scaling data with outliers
## Centering kernel matrices
