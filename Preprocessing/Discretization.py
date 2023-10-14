#K-BIN  discretization
X = np.array([[],
              [],
              []])
est = preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode='ordinal').fit(X)
est.transform(X)  
import pandas as pd
import numpy as np
bins = [0, 1, 13, 20, 60, np.inf]
labels = ['infant', 'kid', 'teen', 'adult', 'senior citizen']
transformer = preprocessing.FunctionTransformer(
  pd.cut, kw_args={'bins': bins, 'labels': labels, 'retbins': False})
transformer.fit_transform(X)

# Feature binarization
binarizer = preprocessing.Binarizer().fit(X)
  
