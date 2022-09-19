import pandas as pd
import seaborn as sn  # for plotting purposes
import matplotlib.pyplot as plt
import numpy as np


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
        distances = [(np.sqrt(np.sum((np.array(x_train.iloc[i]) - np.array(row[1]))) ** 2)) for i in
                     range(len(x_train))]  # Euclidean distance from all other points
        distances =np.array(distances)
        labels = y_train[np.argpartition(distances,k)[0:k]].tolist() # labels of the k nearest neighbors
        predicted_label = max(set(labels),key=labels.count)
        predicted.append(predicted_label)
    return predicted


K = 5
y_pred = knn_classify(K, X_train, y_train, X_test)

data = {'y_Actual': y_test,
        'y_Predicted': y_pred
        }
df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
# calculate confusion matrix and accuracy
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
CM = confusion_matrix.values.tolist()
accuracy = (CM[1][1] + CM[0][0] + CM[2][2]) / (
        sum(CM[1]) + sum(CM[0]) + sum(CM[2]))

print("Accuracy of prediction is equal to: ", accuracy * 100, "%")
sn.heatmap(confusion_matrix, annot=True, cmap="cool", linewidths=1, fmt='g')
plt.show()
