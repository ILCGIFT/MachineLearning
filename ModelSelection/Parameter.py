from sklearn.utils.fixes import loguniform
{'C': loguniform(1e0, 1e3),
 'gamma': loguniform(1e-4, 1e-3),
 'kernel': ['rbf'],
 'class_weight':['balanced', None]}

# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv  # noqa
# now you can import normally from model_selection
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV

#Choosing a resource
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
import pandas as pd
param_grid = {'max_depth': [3, 5, 10],
              'min_samples_split': [2, 5, 10]}
base_estimator = RandomForestClassifier(random_state=0)
X, y = make_classification(n_samples=1000, random_state=0)
sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                         factor=2, resource='n_estimators',
                         max_resources=30).fit(X, y)
sh.best_estimator_

#Exhausting the available resources
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
import pandas as pd
param_grid= {'kernel': ('linear', 'rbf'),
             'C': [1, 10, 100]}
base_estimator = SVC(gamma='scale')
X, y = make_classification(n_samples=1000)
sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                         factor=2, min_resources=20).fit(X, y)
sh.n_resources_

#Aggressive elimination of candidates
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
import pandas as pd
param_grid = {'kernel': ('linear', 'rbf'),
              'C': [1, 10, 100]}
base_estimator = SVC(gamma='scale')
X, y = make_classification(n_samples=1000)
sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                         factor=2, max_resources=40,
                         aggressive_elimination=False).fit(X, y)
sh.n_resources_
sh.n_candidates_

#Composite estimators and parameter spaces
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
X, y = make_moons()
calibrated_forest = CalibratedClassifierCV(
   estimator=RandomForestClassifier(n_estimators=10))
param_grid = {
   'estimator__max_depth': [2, 4, 6, 8]}
search = GridSearchCV(calibrated_forest, param_grid, cv=5)
search.fit(X, y)

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
pipe = Pipeline([
   ('select', SelectKBest()),
   ('model', calibrated_forest)])
param_grid = {
   'select__k': [1, 2],
   'model__estimator__max_depth': [2, 4, 6, 8]}
search = GridSearchCV(pipe, param_grid, cv=5).fit(X, y)
