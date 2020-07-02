import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,make_scorer


# 数据预处理
def data_prep(fpath):
    data = pd.read_csv(fpath)
    data['division'] = data['division'].replace(['accounting','hr','IT','management','marketing','product_mng','RandD','sales','support','technical'],[1,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000])
    data['package'] = data['package'].replace(['a', 'b', 'c', 'd', 'e'], [0, 10, 20, 30, 40])
    data['salary'] = data['salary'].replace(['low', 'medium', 'high'], [0, 50, 100])
    # data.to_csv("data/clean.csv", index=False)
    return data.to_numpy()


if __name__ == "__main__":
	#训练集数据4：1比例拆分成训练和测试
    seed = 42
    test_size = 0.2
    data_path = "train.csv"
    data_path1 = "test.csv"

    data = data_prep(data_path)
    test_data = data_prep(data_path1)
    X = data[:, 1:-1]
    Y = data[:, -1]
    X_train, X_test, y_train,y_test = train_test_split(X, Y,
                                                        test_size=test_size,
                                                        random_state=seed)

    # X_train = X
    # y_train = Y
    # X_test = test_data[:, 1:]
    # ids = test_data[:, 0]


    #建立adaboost模型
    DTR = DecisionTreeRegressor()
    bdt = AdaBoostRegressor(base_estimator= DTR)

    #cross validation 
    my_scorer = make_scorer(mean_squared_error)
    grid_clf_acc = GridSearchCV(estimator=bdt,
             param_grid={"base_estimator__max_depth":[13],
             			"n_estimators":[2000,3000],
             			"learning_rate":[0.01],
             			"loss": ["square"]
             			},
             			scoring =my_scorer)

    grid_clf_acc.fit(X_train, y_train)
    y_pred_acc = grid_clf_acc.predict(X_test)

# New Model Evaluation metrics 
#输出结果
    error = mean_squared_error(y_test, y_pred_acc)
    print(error)
    print(pd.DataFrame(grid_clf_acc.cv_results_))
    outfile = open("adaresult.csv",'w')
    pd.DataFrame(grid_clf_acc.cv_results_).to_csv(outfile)
    outfile.close()

    # param1 = {
    #     'verbosity': 0,
    #     'max_depth': 13,
    #     'min_child_weight': 1,
    #     'learning_rate': 0.027,
    #     'gamma': 0.09,
    #     'reg_lambda': 0.3,
    #     'colsample_bytree':0.9,
    #     'subsample':0.9,
    #     'num_round':10000


    # }
    # model1 = xgboost.XGBRegressor(**param1)
    # model1.fit(X_train, y_train)
    # y_pred1 = model1.predict(X_test)
    # error1 = mean_squared_error(y_test, y_pred1)

    # print(error1)