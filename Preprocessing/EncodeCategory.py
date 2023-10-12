from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
enc = preprocessing.OrdinalEncoder()
X = [[],
     []]
enc.fit(X)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
enc = Pipeline(steps=[("encoder", preprocessing.OrdinalEncoder()),
                      ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
                    ])
enc.fit_transform(X)
## similar to OnehotEncoder

## Infrequent Category
enc = preprocessing.OrdinalEncoder(min_frequency=6).fit(X)
enc.infrequent_categories_
enc.transform(np.array([[],
                        [],
                        []
                        ]))
enc = preprocessing.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=3,
                                   max_categories=3, encoded_missing_value=4)


## Target Encoder

