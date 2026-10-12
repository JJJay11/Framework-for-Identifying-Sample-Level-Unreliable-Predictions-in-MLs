from sklearn.semi_supervised import LabelPropagation,LabelSpreading,SelfTrainingClassifier
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve,auc
import random


#####################
# python3.6.5
# Framework for Identifying Sample-Level Unreliable Predictions in MLs
#####################


# Cuda Device
device = 'cuda'

# Landslide Contribiting Factors
save_columns = ["DRI","TWI","DGL",'Lithology',"Altitude","Slope","Aspect","Qulv_ping","Qulv_pou","Undulation","NDVI","DRO","DSE"]

# Landslide data
ZQ_point = pd.read_csv(u".\data_1.txt",sep=",",header=0,encoding="utf-8")
data_1 = ZQ_point.iloc[:,:]
for i in data_1.columns:
    if i not in save_columns:
        del data_1[i]
x_1 = np.array(data_1)
y_1 = np.ones((len(x_1),))

# Non-landslide data
MRD_point = pd.read_csv(u".\data_0.txt",sep=",",header=0,encoding="utf-8")
data_0 = MRD_point.iloc[:,:]
for i in data_0.columns:
    if i not in save_columns:
        del data_0[i]
x_0 = np.array(data_0)
y_0 = np.zeros((len(x_0),))

# All data
ALL_point = pd.read_csv(u".\ALL.txt",sep=",",header=0,encoding="utf-8")
ALL = ALL_point.iloc[:,:]
for i in ALL.columns:
    if i not in save_columns:
        del ALL[i]
ALL = np.array(ALL)
ALL_0 = np.zeros((len(ALL),))

# Standard normolization
scaler = StandardScaler()
scaler.fit(ALL)
x_1d = scaler.transform(x_1)
x_0d = scaler.transform(x_0)
data_all_d = scaler.transform(ALL)

# Sample definition
X = x_0d
y = y_0

# M-distance
mean_vec = np.mean(x_1d, axis=0)
cov_matrix = np.cov(x_1d, rowvar=False)
inv_cov_matrix = np.linalg.inv(cov_matrix)

def mahalanobis_distance(point, mean_vec, inv_cov_matrix):
    diff = point - mean_vec
    distance = np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff.T))
    return distance

# M-distance result
distances = pd.DataFrame(np.array([mahalanobis_distance(i, mean_vec, inv_cov_matrix) for i in data_all_d]))

# M-distance result output
# distances.to_csv(r"C:\Users\kk\PycharmProjects\Arcgis_FBJ\LSM\00\M_distance_result_output.txt",index =False,header=True)

# Bootstrap Definition
def random_indices(arr, num_rows):
    indices = np.random.choice(arr.shape[0], num_rows, replace=True)
    return indices


DATA_PRED = None
DATA_PRED_test = None
mode = "MLs"

train_accuracy = []
train_precision = []
train_recall = []
train_f1 = []
test_accuracy = []
test_precision = []
test_recall = []
test_f1 = []

dict = {}

for i in tqdm(range(5000)):
    # bootstrap sampling
    random_index_non_landslide = random_indices(X,len(x_1d))
    print("num of landslides: ",len(x_1d))
    print("num of non-landslides: ",len(y[random_index_non_landslide]))

    X_1 = x_1d
    Y_1 = np.ones((len(X_1)),)
    X_0 = X[random_index_non_landslide]
    Y_0 = np.zeros((len(X_0)),)

    train_size = 0.7
    idx_1 = np.arange(X_1.shape[0])
    train_idx_1 = np.random.choice(idx_1, size=round(X_1.shape[0]*train_size), replace=True)
    test_idx_1 = np.setdiff1d(idx_1, train_idx_1)

    idx_0 = np.arange(X_0.shape[0])
    train_idx_0 = np.random.choice(idx_0, size=round(X_0.shape[0]*train_size), replace=True)
    test_idx_0 = np.setdiff1d(idx_0, train_idx_0)

    x_train, y_train = zip(*np.random.permutation(list(zip(np.concatenate((X_1[train_idx_1],X_0[train_idx_0]), axis=0),np.concatenate((Y_1[train_idx_1],Y_0[train_idx_0]), axis=0)))))
    x_test, y_test = zip(*np.random.permutation(list(zip(np.concatenate((X_1[test_idx_1],X_0[test_idx_0]), axis=0),np.concatenate((Y_1[test_idx_1],Y_0[test_idx_0]), axis=0)))))
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


    if mode == "MLs":
        # Model training
        # rf = MLPClassifier(hidden_layer_sizes=(random.randint(50,100),))
        # rf = LogisticRegression(C=random.uniform(0.5,1.0))
        rf = SVC(kernel='rbf',cache_size=5000,probability=True,C=random.uniform(0.05,0.15),gamma=random.randint(5,15))
        rf.fit(x_train, y_train)

        # Model predict
        y_pred = rf.predict(x_test)
        y_pred_proba = rf.predict_proba(x_test)[:,1]
        y_pred = np.where(y_pred <= 0.5, 0, 1)
        y_test = np.where(y_test <= 0.5, 0, 1)

        for sample_idx in test_idx_1:
            if sample_idx not in dict:
                dict[sample_idx] = []
                dict[sample_idx].append(rf.predict_proba(X_1[sample_idx].reshape(1,-1,))[:,1])  # 放入概率值
            else:
                dict[sample_idx].append(rf.predict_proba(X_1[sample_idx].reshape(1,-1,))[:,1])  # 放入概率值

        if i == 0:
            DATA_PRED_test = pd.DataFrame(y_pred_proba)
        else:
            DATA_PRED_test = pd.concat((DATA_PRED_test,pd.DataFrame(y_pred_proba)),axis=1)

        data_pred = rf.predict_proba(data_all_d)[:,1]

        if i == 0:
            DATA_PRED = pd.DataFrame(data_pred)
        else:
            DATA_PRED = pd.concat((DATA_PRED,pd.DataFrame(data_pred)),axis=1)


Dict = {}
for sample_idx in dict.keys():
    Dict[sample_idx] = list(np.array(pd.Series(dict[sample_idx]).quantile(q=0.975)-pd.Series(dict[sample_idx]).quantile(q=0.025)))[0]

# Statistical result
plt.hist(Dict.values(),bins=100)
plt.show()

print("lower thereshold",pd.DataFrame(Dict.values()).quantile(q=0.025)[0])
print("upper thereshold",pd.DataFrame(Dict.values()).quantile(q=0.975)[0])

pu_result = pd.DataFrame(DATA_PRED.quantile(q=0.975,axis=1)-DATA_PRED.quantile(q=0.025,axis=1))
pu_result.columns = ["PU"]

# Result output
kk = pd.concat([ALL_point,pu_result],axis=1)
# kk.to_csv(r"C:\Users\kk\PycharmProjects\Arcgis_FBJ\LSM\00\sample_pred_variability_result.txt",index =False,header=True)

