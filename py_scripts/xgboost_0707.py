import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import LabelEncoder



def data_prep(fpath):
    data = pd.read_csv(fpath)
    # print(data.groupby("salary")["satisfaction_level"].mean())
    # print(data.groupby("division")["satisfaction_level"].mean())
    # print(data.groupby("package")["satisfaction_level"].mean())
    # a = data.groupby("division")["satisfaction_level"].mean()

    data['salary'] = data['salary'].replace(["low","medium","high"],[0.602128,0.622538,0.638931])
    data['division'] = data['division'].replace(['accounting','hr','IT','management','marketing','product_mng','RandD','sales','support','technical'],
        [0.585789,0.598847,0.618778,0.629205,0.617997,0.617918,0.616388,0.614893,0.617225,0.612363])
    data['package'] = data['package'].replace(['a', 'b', 'c', 'd', 'e'], [0.665958, 0.433325,0.663722,0.446261,0.675485])
    
  
    return data.to_numpy()

    print(data)


if __name__ == "__main__":
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

    param1 = {
        'verbosity': 0,
        'max_depth': 11,
        'min_child_weight': 1,
        'learning_rate': 0.0319,
        'gamma': 0.09,
        'reg_lambda': 0.26,
        'colsample_bytree':0.9,
        'subsample':0.9,
        'n_estimators':100
    }

    model1 = xgboost.XGBRegressor(**param1)
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)
    error1 = mean_squared_error(y_test, y_pred1)

    print(error1)

    # result = np.c_[ids, y_pred1]
    # outfile = "result1.csv"
    # df = pd.DataFrame(result,columns=['id','satisfaction_level'])
    # df = df.astype({'id': 'int32'})
    # df.to_csv(outfile,index=None)

    bdt = AdaBoostRegressor(DecisionTreeRegressor(max_depth=13),
                                                        n_estimators=100, 
                                                        learning_rate=0.03,
                                                        loss = "square")

    bdt.fit(X_train, y_train)
    y_pred2 = bdt.predict(X_test)
    error2 = mean_squared_error(y_test, y_pred2)
    print(error2)

    error_total = mean_squared_error(y_test, (y_pred1+y_pred2)/2)
    print(error_total)


    

# # New Model Evaluation metrics 

