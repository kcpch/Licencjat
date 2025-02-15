import numpy as np
import pandas as pd
from typing import Tuple
import xgboost as xgb
from rope.base.oi.type_hinting.utils import parametrize_type
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def new_loss_function(lambda_, patrial_dependence_profile, variables, function = 'sqrt'):
    def gradient(predt: np.ndarray, dtrain: xgb.DMatrix, lambda_: float, f: np.ndarray) -> np.ndarray:
        y = dtrain.get_label()
        # print(len(f.columns))
        copy = f.copy()
        copy['predt'] = predt
        print(copy)
        result = [np.sum((predt-np.array(f.iloc[:,i]))**2) for i in range(len(f.columns))]
        print(result)
        print('PDP distance:', np.sum(result))
        # result = np.sum(result)
        grad_results = [2 * (predt-np.array(f.iloc[:,i])) for i in range(2)]
        return 2 * (predt - y) + lambda_ * np.sum(grad_results, axis=0)

    def hessian(predt: np.ndarray, dtrain: xgb.DMatrix, lambda_: float, f: np.ndarray) -> np.ndarray:
        result = 2 + 2 * lambda_
        if type(result) in [int, float]:
            result = result * np.ones_like(dtrain.get_label())
        return result

    def gradient_KL(predt: np.ndarray, dtrain: xgb.DMatrix, lambda_: float, f: np.ndarray) -> np.ndarray:
        print('Distance between PDP:', np.sum(predt * np.log2(predt/f)))
        y = dtrain.get_label()
        return 2 * (predt - y) + lambda_ * np.log2(predt/f) + 1/np.log(2)

    def hessian_KL(predt: np.ndarray, dtrain: xgb.DMatrix, lambda_: float, f: np.ndarray) -> np.ndarray:
        y = dtrain.get_label()
        return 2 + lambda_ * f/(predt * np.log(2))

    def obj_sq(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        X_train = pd.DataFrame(dtrain.get_data().toarray())
        X_train.columns = dtrain.feature_names
        # y_train = dtrain.get_label()


        for i in range(2):
            variable = variables[i]
            pdp = patrial_dependence_profile.iloc[np.where(patrial_dependence_profile.variable == variable)[0]]

            plt.scatter(X_train[variable], predt, c='r', s=1)
            plt.scatter(pdp['x'], pdp['yhat'], c='b', s=1)
            plt.legend(['train', 'pdp'])
            plt.xlabel(variable)
            plt.title('Distance between PDP: variable:'+str(variable))
            plt.show()

            X_train['x'] = X_train[variable]
            X_train['interpolated'] = True
            pdp['interpolated'] = False
            X = pd.concat([pdp[['x', 'yhat', 'interpolated']], X_train[['x', 'interpolated']]])


            X['yhat'] = X['yhat'].interpolate(method='linear')

            # plt.scatter(X[X['interpolated'] == False]['x'], X[X['interpolated'] == False]['yhat'], c='b', label='pdp')
            # plt.scatter(X[X['interpolated'] == True]['x'], X[X['interpolated'] == True]['yhat'], c='r', label='interpolated', s=5)
            #
            # plt.legend()
            # plt.show()

            X = X[X['interpolated'] == True]
            try:
                result[variable] = X['yhat'].values
            except:
                result = X[['yhat']]
                result.columns = [variable]


        if function == 'sqrt':
            grad = gradient(predt, dtrain, lambda_, result)
            hess = hessian(predt, dtrain, lambda_, result)
        elif function == 'kl':
            grad = gradient_KL(predt, dtrain, lambda_, result)
            hess = hessian_KL(predt, dtrain, lambda_, result)
        else:
            raise Exception('error')

        return grad, hess

    return obj_sq

def xgboost_new(X, y, pdp, lambda_, variables, func = 'sqrt'):

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
        num_boost_round=2,
        obj=new_loss_function(lambda_, pdp, variables, func),
        evals=[(dtrain, 'train'), (dtest, 'test')]
    )

    # print('result', new_loss_function(lambda_, pdp, variables, func)[1])

    return model