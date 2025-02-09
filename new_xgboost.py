import numpy as np
import pandas as pd
from typing import Tuple
import xgboost as xgb
from sklearn.model_selection import train_test_split

import preprocessing

import matplotlib.pyplot as plt


def new_loss_function(lambda_, pdp):
    def gradient(predt: np.ndarray, dtrain: xgb.DMatrix, lambda_: float, f: np.ndarray) -> np.ndarray:
        y = dtrain.get_label()
        return 2 * (predt - y) + 2 * lambda_ * (predt - f)

    def hessian(predt: np.ndarray, dtrain: xgb.DMatrix, lambda_: float, f: np.ndarray) -> np.ndarray:
        result = 2 + 2 * lambda_
        if type(result) in [int, float]:
            result = result * np.ones_like(dtrain.get_label())
        return result

    def obj_sq(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        X_train = pd.DataFrame(dtrain.get_data().toarray())
        X_train.columns = dtrain.feature_names
        # y_train = dtrain.get_label()

        plt.scatter(X_train['num__age'], predt, c='r', s=1)
        plt.scatter(pdp['x'], pdp['yhat'], c='b', s=1)
        plt.legend(['train', 'pdp'])
        plt.show()

        X_train['x'] = X_train['num__age']
        X_train['interpolated'] = True
        pdp['interpolated'] = False
        X = pd.concat([pdp[['x', 'yhat', 'interpolated']], X_train[['x', 'interpolated']]])
        X = X.sort_values(by='x')

        X['yhat'] = X['yhat'].interpolate(method='linear')

        plt.scatter(X[X['interpolated'] == False]['x'], X[X['interpolated'] == False]['yhat'], c='b', label='pdp')
        plt.scatter(X[X['interpolated'] == True]['x'], X[X['interpolated'] == True]['yhat'], c='r', label='interpolated', s=5)

        plt.legend()
        plt.show()

        print('X_train shape:', X_train.shape)
        print('pdp shape:', pdp.shape)
        print('X shape:', X.shape)

        X_train = X[X['interpolated'] == True]['yhat'].values

        print('X_train shape:', X_train.shape)
        print(X_train)

        grad = gradient(predt, dtrain, lambda_, X_train)
        hess = hessian(predt, dtrain, lambda_, X_train)
        if type(hess) in [int, float]:
            hess = hess * np.ones_like(grad)
        return grad, hess

    return obj_sq

def xgboost_new(X, y, pdp, lambda_):
    # X = preprocessing.trainer(X)

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
        num_boost_round=5,
        obj=new_loss_function(lambda_, pdp.iloc[np.where(pdp.variable == 'num__age')[0]]),
        evals=[(dtrain, 'train'), (dtest, 'test')]
    )

    return model