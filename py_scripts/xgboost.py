import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error


def data_prep(fpath):
    data = pd.read_csv(fpath)
    data = data.drop(columns=['division'])
    data['package'] = data['package'].replace(['a', 'b', 'c', 'd', 'e'], [0, 1, 2, 3, 4])
    data['salary'] = data['salary'].replace(['low', 'medium', 'high'], [0, 1, 2])
    # data.to_csv("data/clean.csv", index=False)
    return data.to_numpy()


if __name__ == "__main__":
    seed = 42
    test_size = 0.2
    data_path = "data/train.csv"

    data = data_prep(data_path)
    X = data[:, :-1]
    Y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=test_size,
                                                        random_state=seed)

    param = {
        'verbosity': 0,
        'max_depth': 6,
        'min_child_weight': 1,
        'learning_rate': 0.3,
        'gamma': 0,
        'reg_lambda': 1,
    }
    model = xgboost.XGBRegressor(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    error = mean_squared_error(y_test, y_pred)

    print('MSE:', error)
