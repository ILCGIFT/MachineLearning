from sklearn.preprocessing import normalize,Normalizer

X = [[],
     [],
     []]
X_normalized = preprocessing.normalize(X, norm='l2')
normalizer = preprocessing.Normalizer().fit(X)
normalizer.transform(X)
