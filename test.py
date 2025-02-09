import preprocessing
import new_xgboost

import dalex as dx
titanic = dx.datasets.load_titanic()
X = titanic.drop(columns='survived')
y = titanic.survived

from sklearn.tree import DecisionTreeClassifier

X = preprocessing.trainer(X)

model = preprocessing.trainer(X, y, DecisionTreeClassifier(max_depth=4))
pdp = preprocessing.make_profile(X, y, model)

model = new_xgboost.xgboost_new(X, y, pdp, 20)

