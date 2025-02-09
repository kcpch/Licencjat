import pandas as pd
import dalex as dx
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def trainer(X, y = None, model = None):
    """
    Funkcja do trenowania wybranego modelu wraz z preprocessingiem
    :param X:
    :param y:
    :param model:
    :return: wytrenowany model
    """

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
    if model is not None:
        model_to_train = Pipeline(
            steps=[
                ('preprocessing', preprocessor),
                ('model', model)
            ]
        )
        model_to_train.fit(X, y)
        return model_to_train
    else:
        result = pd.DataFrame(preprocessor.fit_transform(X, y))
        result.columns = preprocessor.get_feature_names_out()
        return result



def make_profile(X, y, model):
    """
    Funcja do tworzenia profili pdp
    :param X:
    :param y:
    :param model: wytrenowany model (można użyć funkcji trainer())
    :return: df z kolumnami ['variable', 'x', 'yhat']
    """
    exp = dx.Explainer(model, X, y, verbose=False)
    # print(X.columns.tolist())
    pdp = exp.model_profile(variables = X.columns.tolist(), verbose=False)
    result = pdp.result[['_vname_', '_x_', '_yhat_']]
    result.columns = [['variable', 'x', 'yhat']]
    return result
