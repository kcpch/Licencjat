import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import matplotlib.pyplot as plt


def new_loss_function(lambda_, pdp):
    def gradient(predt: np.ndarray, dtrain: xgb.DMatrix, lambda_: float, f: np.ndarray) -> np.ndarray:
        y = dtrain.get_label()
        return 2 * (predt - y) + 2 * lambda_ * (predt - f)

    def hessian(predt: np.ndarray, dtrain: xgb.DMatrix, lambda_: float, f: np.ndarray) -> np.ndarray:
        # y = dtrain.get_label()
        return 2 + 2 * lambda_

    def obj_sq(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        X_train = pd.DataFrame(dtrain.get_data().toarray())
        X_train.columns = dtrain.feature_names
        y_train = dtrain.get_label()

        # f = pdp_results[['_x_', '_yhat_']] # teraz to jest pdp
        # f.columns = ['x', 'yhat']

        print(X_train)
        print(y_train)

        plt.scatter(X_train['num__age'], predt, c='r', s=1)
        plt.scatter(f['x'], f['yhat'], c='b', s=1)
        plt.show()

        df['x'] = df['num__age']
        df['interpolated'] = True
        f['interpolated'] = False
        combined_df = pd.concat([f[['x', 'yhat', 'interpolated']], df[['x', 'interpolated']]])
        combined_df = combined_df.sort_values(by='x')

        display(combined_df)
        combined_df['yhat'] = combined_df['yhat'].interpolate(method='linear')

        display(combined_df)

        plt.scatter(combined_df['x'], combined_df['yhat'], c=combined_df['interpolated'])
        plt.show()

        print('df shape:', df.shape)
        print('f shape:', f.shape)
        print('combined_df shape:', combined_df.shape)

        f = combined_df[combined_df['interpolated'] == True]['yhat'].values

        print('f shape:', f.shape)

        lambda_ = 100
        grad = gradient(predt, dtrain, lambda_, f)
        hess = hessian(predt, dtrain, lambda_, f)
        if type(hess) in [int, float]:
            hess = hess * np.ones_like(grad)
        return grad, hess

    return obj_sq

# new_loss_function(1,5)


# from sklearn.metrics import roc_auc_score
# print(roc_auc_score(model.predict(X), y))

# df = pd.DataFrame(dtrain.get_data().toarray())
# df.columns = dtrain.feature_names
# df
# # plt.scatter(df['num__age'], np.ones_like(df['num__age']), c='r', s=1)
# # plt.scatter(f['_x_'], f['_yhat_'], c='b', s=1)

# df['x'] = df['num__age']
# df['interpolated'] = True
# f['interpolated'] = False
# f['x'] = f['_x_']
# combined_df = pd.concat([f[['x', '_yhat_', 'interpolated']], df[['x', 'interpolated']]])
# combined_df = combined_df.sort_values(by='x')

# display(combined_df)
# combined_df['_yhat_'] = combined_df['_yhat_'].interpolate(method='linear')

# display(combined_df)

# plt.scatter(combined_df['x'], combined_df['_yhat_'], c=combined_df['interpolated'])
# plt.show()