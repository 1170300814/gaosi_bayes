# 科学计算包
import numpy as np
# 画图包
from matplotlib import pyplot as plt
# 封装好的KMeans聚类包
from sklearn.cluster import KMeans
import pandas as pd
from numpy.linalg import inv
from numpy import dot
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from  matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # 生成两个个二维的矩阵
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)  # .T是求转置
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y=X[y == c1, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=c1)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')


juleidata = pd.read_csv('001.csv')
data=juleidata.iloc[:,1:3]
xtrain=juleidata.iloc[:5801,1:3]
xtest=juleidata.iloc[5800:6000,1:3]
# xtrain['x0']=1
data1=data
data2=data1.values
print(data2)

model1 = KMeans(n_clusters=3)

model1.fit(data)

C_i = model1.predict(data)
ytrain=[]
ytest=[]
for i in range(5801):
    ytrain.append(C_i[i])
for i in range(5800,6000):
    ytest.append(C_i[i])
logreg = LogisticRegression().fit(xtrain, ytrain)
print("Training set score: {:.5f}".format(logreg.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(logreg.score(xtest, ytest)))

sc = StandardScaler()
sc.fit(xtrain)
X_train_std = sc.transform(xtrain)
X_test_std = sc.transform(xtest)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((ytrain, ytest))
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, ytrain)
lr.predict_proba([X_test_std[0,:]])
plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105,150))
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend(loc='upper left')
plt.show()
theta_n = dot(dot(inv(dot(xtrain.T, xtrain)), xtrain.T), ytrain)

rightnum=0

for i in range(5800,6000):
    score=theta_n[0]*xtest['V1'][i]+theta_n[1]*xtest['V2'][i]+theta_n[2]
    if round(score)==ytest[i-5800]:
        rightnum+=1
print(theta_n)
print("rightratio is : ",int((rightnum/200)*100),'%')
# # 还需要知道聚类中心的坐标
Muk = model1.cluster_centers_

# 画图
plt.scatter(data2[:,0],data2[:,1],c=C_i,cmap=plt.cm.Paired)
# 画聚类中心
plt.scatter(Muk[:,0],Muk[:,1],marker='*',s=60)
for i in range(3):
    plt.annotate('中心'+str(i + 1),(Muk[i,0],Muk[i,1]))
plt.show()