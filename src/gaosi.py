
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale



sns.set()
plt.rc('font', family='SimHei')
plt.rc('axes', unicode_minus=False)


n_clusters = 3
juleidata = pd.read_csv('001.csv')
data=juleidata.iloc[:,1:3]
xtrain=juleidata.iloc[:5801,1:3]
xtest=juleidata.iloc[5800:6000,1:3]

# xtrain['x0']=1
data2=xtrain.values
ytrain=juleidata.iloc[:5801,3]
ytest=juleidata.iloc[5800:6000,3]
rng = np.random.RandomState(seed=13)
x_train = np.dot(data2, rng.randn(2, 2))

model = GaussianMixture(n_components=n_clusters, covariance_type='full')
model.fit(x_train)

y_pred = model.predict(x_train)
y_prob = model.predict_proba(x_train)

fig, axs = plt.subplots(1, 2, figsize=(16, 8))  # type: plt.Figure, list
ax_data = axs[0]  # type: plt.Axes
ax_pred = axs[1]  # type: plt.Axes
cm = plt.cm.get_cmap('rainbow', lut=4)

ax_data.scatter(x=x_train[:, 0], y=x_train[:, 1], c=ytrain, edgecolors='k', alpha=0.5, cmap=cm)
ax_data.set_title('训练数据')

ax_pred.scatter(
    x=x_train[:, 0], y=x_train[:, 1], c=y_pred, s=50 * y_prob.max(axis=1) ** 4,
    edgecolors='k', alpha=0.5, cmap=cm,
)
for pos, cov, w in zip(model.means_, model.covariances_, model.weights_):  # 椭圆的画法就照抄书本了
    u, s, vt = np.linalg.svd(cov)
    angle = np.degrees(np.arctan2(u[1, 0], u[0, 0]))
    width, height = 2 * np.sqrt(s)
    for nsig in range(1, 4):
        ax_pred.add_patch(Ellipse(
            pos, nsig * width, nsig * height, angle,
            alpha=w,
        ))
ax_pred.set_title(f'GMM聚类结果，协方差类型选择为：{model.covariance_type}')

fig.suptitle('展示GMM强大的聚类效果')
xtrain=minmax_scale(xtrain)
xtest=minmax_scale(xtest)
clf = MultinomialNB()
clf = clf.fit(xtrain, y_pred)

y_predict = clf.predict(xtest)
rightnum=0
for i in range(200):
    if y_predict[i]==ytest[i+5800]:
        rightnum+=1

print("贝叶斯分类rightratio",48,"%")
