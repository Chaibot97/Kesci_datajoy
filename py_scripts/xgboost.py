'''
    One params works well for now.
    To prevent being forgotten after changing, I upload it with the methods of adjusting parameters.
    It's weird that the model get better score by dropping the column'dividision'.
'''
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error


def data_prep(fpath):
    data = pd.read_csv(fpath)
    data = data.drop(columns=['division'])
#    data['division'] = data['division'].replace(
#        ['accounting', 'hr', 'IT', 'management', 'marketing', 'product_mng', 'RandD', 'sales', 'support', 'technical'],
#        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data['package'] = data['package'].replace(['a', 'b', 'c', 'd', 'e'], [0, 1, 2, 3, 4])
    data['salary'] = data['salary'].replace(['low', 'medium', 'high'], [0, 1, 2])
    return data.to_numpy()


if __name__ == "__main__":
    seed = 42
    test_size = 0.2
    data_path = "data/train.csv"

    data = data_prep(data_path)
    X = data[:, 1:-1]
    Y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=test_size,
                                                        random_state=seed)

    param = {
        'learning_rate': 0.05, 'n_estimators': 200, 'max_depth': 9, 'min_child_weight': 6, 'seed': 0,
        'subsample': 0.9, 'colsample_bytree': 0.8, 'gamma': 0.16, 'reg_alpha': 0.1, 'reg_lambda': 0.1
    }

    """
    test_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 80, 'max_depth': 6, 'min_child_weight': 3, 'seed': 0,
                'subsample': 0.9, 'colsample_bytree': 0.8, 'gamma': 0.1, 'reg_alpha': 1, 'reg_lambda': 0.1}
    
        grid = GridSearchCV(clf1, param_dist, cv=3, scoring='neg_log_loss', n_jobs=-1)


    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=3, verbose=1, n_jobs=1)
    optimized_GBM.fit(X_train, y_train)
    print('the best params：{0}'.format(optimized_GBM.best_params_))
    print('the best model scores:{0}'.format(optimized_GBM.best_score_))
    """
    model = xgboost.XGBRegressor(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    error = mean_squared_error(y_test, y_pred)

    print('MSE:', error)
