import xgboost as xgb
from matplotlib import pyplot as plt
# read in data
dtrain = xgb.DMatrix("train_10000.txt")
dtest = xgb.DMatrix("train_2000.txt")
# specify parameters via map
param = {'max_depth':6, 'eta':0.01, 'verbosity':2,'alpha':0.5}
num_round = 20000
bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dtest)

config = bst.save_config()

origin = []

f = open("train_2000.txt")
for line in f:
	origin.append(float((line.split())[0]))

f.close()

result = 0;
for i in range(0,len(preds)):
	result+= (preds[i]-origin[i])**2

result = result/len(preds)
print(result)

f1 = open("config.txt","w")
f1.write(config)
f1.close()

xgb.plot_importance(bst)
plt.show()