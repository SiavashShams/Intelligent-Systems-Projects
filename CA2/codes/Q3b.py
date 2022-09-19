from metric_learn import LMNN
from metric_learn import NCA
import pandas as pd
import seaborn as sn  # for plotting purposes
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import mode

df = pd.read_csv("wine.csv")  # load data
print(df.columns)
# split data into training and test sets
training_data = df.sample(frac=0.8, random_state=0)
X_train = training_data.drop("Class", axis=1).reset_index(drop=True)
y_train = training_data["Class"].reset_index(drop=True)
testing_data = df.drop(training_data.index)
X_test = testing_data.drop("Class", axis=1).reset_index(drop=True)
y_test = testing_data["Class"].reset_index(drop=True)

# Function to calculate KNN
def knn_classify(k, x_train, y_train, x_test):
    predicted = []
    for row in x_test.iterrows():
        distances = [np.sqrt(np.sum((np.array(x_train.iloc[i]) - np.array(row[1]))) ** 2) for i in
                     x_train.index]  # Euclidean distance from all other points
        distances = np.array(distances)
        labels = y_train[np.argpartition(distances,k)[0:k]].tolist() # labels of the k nearest neighbors
        predicted_label = max(set(labels), key=labels.count)
        predicted.append(predicted_label)
    return predicted
# you can change K in order to see other plots
K=5
lmnn = LMNN(max_iter=10000,convergence_tol=1e-4)
nca = NCA(max_iter=10000,tol=1e-4)
lmnn.fit(X_train, y_train)
nca.fit(X_train, y_train)
X_LMNN = pd.DataFrame(lmnn.transform(X_train))
XX_test_LMNN = pd.DataFrame(lmnn.transform(X_test))
X_NCA = pd.DataFrame(nca.transform(X_train))
XX_test_NCA = pd.DataFrame(nca.transform(X_test))
y_pred_LLMN = knn_classify(K, X_LMNN, y_train, XX_test_LMNN)
y_pred_NCA = knn_classify(K, X_NCA, y_train, XX_test_NCA)
data = {'y_Actual': y_test,
        'y_Predicted': y_pred_LLMN
        }
data_NCA = {'y_Actual': y_test,
            'y_Predicted': y_pred_NCA
            }
df_LMNN = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
df_NCA = pd.DataFrame(data_NCA, columns=['y_Actual', 'y_Predicted'])
# calculate confusion matrix and accuracy for each method
confusion_matrix_LMNN = pd.crosstab(df_LMNN['y_Actual'], df_LMNN['y_Predicted'], rownames=['Actual'],
                                    colnames=['Predicted'])
confusion_matrix_NCA = pd.crosstab(df_NCA['y_Actual'], df_NCA['y_Predicted'], rownames=['Actual'],
                                   colnames=['Predicted'])
print(confusion_matrix_LMNN)
print(confusion_matrix_NCA)
CM_LMNN = confusion_matrix_LMNN.values.tolist()
accuracy_LMNN = (CM_LMNN[1][1] + CM_LMNN[0][0] + CM_LMNN[2][2]) / (
        sum(CM_LMNN[1]) + sum(CM_LMNN[0]) + sum(CM_LMNN[2]))

CM_NCA = confusion_matrix_NCA.values.tolist()
accuracy_NCA = (CM_NCA[1][1] + CM_NCA[0][0] + CM_NCA[2][2]) / (
        sum(CM_NCA[1]) + sum(CM_NCA[0]) + sum(CM_NCA[2]))

print("Accuracy of prediction with LMNN metric is equal to: ", accuracy_LMNN * 100, "%")
print("Accuracy of prediction with NCA metric is equal to: ", accuracy_NCA * 100, "%")

fig, axs = plt.subplots(2, sharex=True)
fig.suptitle('LMNN Vs NCA')
sn.heatmap(confusion_matrix_LMNN, annot=True, cmap="cool", linewidths=1, fmt='g', ax=axs[1])
sn.heatmap(confusion_matrix_NCA, annot=True, cmap="cool", linewidths=1, fmt='g', ax=axs[0])
axs[1].set_title("LMNN metric")
axs[0].set_title("NCA metric")

plt.tight_layout()
plt.show()
accuracyyy={"acc NCA":[],"acc LMNN":[]}

#  calculate accuracy of each method for K=1,2,...,9
for K in range(1,10):
    y_pred_LLMN = knn_classify(K, X_LMNN, y_train, XX_test_LMNN)
    y_pred_NCA = knn_classify(K, X_NCA, y_train, XX_test_NCA)
    data = {'y_Actual': y_test,
            'y_Predicted': y_pred_LLMN
            }
    data_NCA = {'y_Actual': y_test,
                'y_Predicted': y_pred_NCA
                }
    df_LMNN = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    df_NCA = pd.DataFrame(data_NCA, columns=['y_Actual', 'y_Predicted'])
    # calculate confusion matrix and accuracy
    confusion_matrix_LMNN = pd.crosstab(df_LMNN['y_Actual'], df_LMNN['y_Predicted'], rownames=['Actual'],
                                        colnames=['Predicted'])
    confusion_matrix_NCA = pd.crosstab(df_NCA['y_Actual'], df_NCA['y_Predicted'], rownames=['Actual'],
                                       colnames=['Predicted'])
    CM_LMNN = confusion_matrix_LMNN.values.tolist()
    accuracy_LMNN = (CM_LMNN[1][1] + CM_LMNN[0][0] + CM_LMNN[2][2]) / (
            sum(CM_LMNN[1]) + sum(CM_LMNN[0]) + sum(CM_LMNN[2]))

    CM_NCA = confusion_matrix_NCA.values.tolist()
    accuracy_NCA = (CM_NCA[1][1] + CM_NCA[0][0] + CM_NCA[2][2]) / (
            sum(CM_NCA[1]) + sum(CM_NCA[0]) + sum(CM_NCA[2]))

    print(f"Accuracy of prediction with LMNN metric with {K} neighbors is equal to: ", accuracy_LMNN * 100, "%")
    print(f"Accuracy of prediction with NCA metric with {K} neighbors  is equal to: ", accuracy_NCA * 100, "%")
    accuracyyy["acc NCA"].append(accuracy_NCA)
    accuracyyy["acc LMNN"].append(accuracy_LMNN)

# plot accuracy Vs K for each method
plt.subplot(2,1,1)
plt.bar(range(1,10),accuracyyy["acc NCA"])
plt.xticks(range(1,10))
plt.title("accuracy of NCA with different K's")
plt.xlabel("K")
plt.ylabel("accuracy")
plt.subplot(2,1,2)
plt.bar(range(1,10),accuracyyy["acc LMNN"])
plt.xticks(range(1,10))
plt.title("accuracy of LMNN with different K's")
plt.xlabel("K")
plt.ylabel("accuracy")
plt.tight_layout()
plt.show()