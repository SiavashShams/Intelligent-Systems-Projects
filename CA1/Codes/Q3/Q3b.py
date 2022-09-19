import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


def make_meshs(x, y):
    #Creating mesh for plot
    xmin = x.min() - 1
    xmax = x.max() + 1
    ymin = y.min() - 1
    ymax = y.max() + 1
    xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.01), np.arange(ymin, ymax, 0.01))
    return xx, yy

#Load data
dataa = datasets.load_iris()
X = dataa.data[:, :2]
y = dataa.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Train and determine Regions
classifier = svm.LinearSVC(max_iter=10000).fit(X, y)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshs(X0, X1)
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#Plot Regions
ax.contourf(xx, yy, Z,cmap=plt.cm.ocean,alpha=0.69)
ax.scatter(X0, X1, c=y, cmap=plt.cm.ocean, s=22, edgecolors="black")
ax.set_ylim(yy.min(), yy.max())
ax.set_xlim(xx.min(), xx.max())
ax.set_ylabel("Sepal width")
ax.set_xlabel("Sepal length")
plt.show()


y_pred = classifier.predict(X_test)
CM = confusion_matrix(y_test, y_pred)
ConfM=confusion_matrix(y_test, y_pred,normalize='true')
print("Confusion Matrix\n",CM)
print("Confidence Matrix\n",ConfM)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.0f} %".format(accuracies.mean()*100))
