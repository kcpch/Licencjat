import preprocessing
import loss_function

import dalex as dx
titanic = dx.datasets.load_titanic()
X = titanic.drop(columns='survived')
y = titanic.survived

from sklearn.tree import DecisionTreeClassifier

model = preprocessing.trainer(X, y, DecisionTreeClassifier(max_depth=4))
df = preprocessing.make_profile(X, y, model)

# print(type(loss_function.new_loss_function(1, df)))

import numpy as np
import pandas as pd

# pd.wide_to_long(df, )
# print(df)
# # df = df.loc[:, df['variable'] == 'age']
# print(df)
# print(np.unique(df['variable']))
# print(df.iloc[np.where(df.variable == 'age')[0]])
# # print(df.pivot_table(index='x', column
# # s='variable', values='yhat'))



# X = pd.DataFrame(preprocessor.fit_transform(X))
# X.columns = preprocessor.get_feature_names_out()

# X = preprocessing.trainer(X)
# print(X.columns)
# print(X)
# print(y)

from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numerical_features = X.columns[(X.dtypes == 'float') | (X.dtypes == 'int64')]
numerical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]
)
categorical_features = X.columns[X.dtypes == 'object']
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
model_to_train = Pipeline(
    steps=[
        ('preprocessing', preprocessor),
        ('model', model)
    ]
)
model_to_train.fit(X, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "max_depth": 5,
   #  "objective": "binary:logistic",
    "eval_metric": "auc"
}

model = xgb.train(
    params,
    dtrain=dtrain,
    num_boost_round=15,
    obj=loss_function.new_loss_function(1, df.iloc[np.where(df.variable == 'age')[0]]),
    evals=[(dtrain, 'train'), (dtest, 'test')]
)
#
# results = pd.DataFrame(model.predict(dtest), y_test).reset_index()
# results.columns = ['true', 'pred']
#
# results['pred_treshold_0.5'] = results['pred']>0.5
# results['pred_treshold_0.5'] = results['pred_treshold_0.5'].astype(int)
#
# results.head(20)
# results
# conf_matrix = confusion_matrix(results['true'], results['pred_treshold_0.5'])
# conf_matrix