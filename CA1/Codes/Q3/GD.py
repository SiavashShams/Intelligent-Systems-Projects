from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Load data
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
#Label the data
for i in range(len(y)):
    if y[i] == 0:
        y[i] = 1
    else:
        y[i] = 0


def Armijo(alpha, z, z_old, sigma=0.01, beta=0.5):
    cur_alpha = alpha
    while z <= z_old + sigma * cur_alpha * (z-z_old):
        cur_alpha *= beta
    return cur_alpha


def GD(alpha0, iters, X, y):
    # Step 1: Initial Parameter
    N = len(X)
    w = np.zeros((X.shape[1], 1))
    b = 0
    for i in range(iters):
        #Apply sigmoid function to get y prediction
        Z = np.dot(w.T, X.T) + b
        y_pred = 1 / (1 + 1 / np.exp(Z))

        #Calculate cost function
        cost = y_pred - y
        """
        if (i > 2):
            Z_old = np.dot((w + alpha0 * dw).T, X.T) + b + alpha0 * db
            alpha0 = Armijo(alpha0, Z, Z_old)
        """
        #Calculate gradient
        dw = 1 / N * np.dot(X.T, (cost).T)
        db = 1 / N * np.sum(cost)

        #Update w and b
        w = w - alpha0 * dw
        b = b - alpha0 * db

    return (w, b)
alpha0=0.01
iters=2000
result=GD(alpha0,iters,X,y)

print("w vector is\n",list(result)[0])
print("b scalar is\n",list(result)[1])
result=list(result)
# Determine line equation
bb=-result[1]/result[0][1]
mm=-result[0][0]/result[0][1]
x0=[0,8]
y0=mm*x0+bb


fig, ax = plt.subplots()
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set2,edgecolor='k')
line=mlines.Line2D(x0,y0,color='g')
ax.add_line(line)
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
plt.show()

ir = pd.DataFrame(X)
ir['Species'] = pd.DataFrame(y)
train=ir.sample(frac=0.8,random_state=0) #random state is a seed value
test=ir.drop(train.index)
test=test.reset_index()
y_pred=[]
#Classify the test data based on the trained model
for i in range(len(test)):
    if test[1][i]<test[0][i]*mm+bb:
        y_pred.append(0)
    else:
        y_pred.append(1)

#Confusion matrix and Confidence Matrix and Accuracy
y_pred=pd.Series(y_pred,name='predicted')
y_actu=pd.Series(test['Species'],name='actual')
CM = pd.crosstab(y_actu, y_pred)
df_conf_norm = CM / CM.sum(axis=1)
accuracy=(CM[0][0]+CM[1][1])/(CM[0][0]+CM[1][1]+CM[0][1]+CM[1][0])
print(CM)
print(df_conf_norm)
print("Accuracy is ",accuracy*100)