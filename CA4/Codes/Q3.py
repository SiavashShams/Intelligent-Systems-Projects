import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

df = pd.read_csv('Surgical.csv')

y = df["complication"]
X = df.drop("complication", axis=1)

# split data to train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_unlabeled, X_train, y_un, y_train = train_test_split(X_train, y_train, test_size=0.013333, random_state=42)
# histogram of distribution of data based on complication
y_train.value_counts().plot(kind='bar')
plt.hist(y_train, rwidth=1, bins=3)
plt.xticks([0, 1], ['Without Complication', 'With Complication'])
plt.ylabel('Number')
plt.show()
# train model
clf = LogisticRegression(max_iter=600)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# confusion matrix and metrics
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt="g")
plt.show()
[precision, recall, f1, _] = precision_recall_fscore_support(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("Accuracy of the model is: ", acc)
print("F1 score of the model is: ", f1)


def self_training(X_train, y_train, X_unlabeled, X_test, y_test):
    f1_iter = []
    num_new_labels = []
    for i in range(len(X_unlabeled)):

        # fit classifier 
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # calculate f1 score in each iteration
        [_, _, f1, _] = precision_recall_fscore_support(y_test, y_pred)
        print("Iteration: ", i)
        print("Test f1: ", f1)
        f1_iter.append(f1)
        # generate probabilities for unlabeled data
        pred_probs = clf.predict_proba(X_unlabeled)
        new_0 = X_unlabeled[pred_probs[:, 0] > 0.99]  # change 0.99 to your desired threshold
        new_1 = X_unlabeled[pred_probs[:, 1] > 0.99]

        num_new_labels.append(len(new_0) + len(new_1))

        # add labeled data to training data
        X_train = pd.concat([X_train, new_0, new_1], axis=0)
        y_train = pd.concat([y_train, pd.Series(np.zeros(len(new_0)), index=new_0.index),
                             pd.Series(np.ones(len(new_1)), index=new_1.index)])

        print("new labels added to training data: ", len(new_0) + len(new_1))
        # drop labeled data
        X_unlabeled = X_unlabeled.drop(index=new_0.index)
        X_unlabeled = X_unlabeled.drop(index=new_1.index)
        # check if there is still data to be labeled
        if len(new_0) + len(new_1) == 0:
            return [X_train, y_train, X_unlabeled, X_test, y_test, i, f1_iter, num_new_labels]
            break


[X_train, y_train, X_unlabeled, X_test, y_test, i, f1_iter, num_new_labels] = self_training(X_train, y_train,
                                                                                            X_unlabeled,
                                                                                            X_test, y_test)
# plot f1 score
f1_iter = np.array(f1_iter)
plt.plot(range(i + 1), f1_iter[:, 1])
plt.ylabel('F1 Score')
plt.xlabel('Iteration')
plt.show()
# plot number of added data in each iteration =
plt.plot(range(i + 1), num_new_labels)
plt.ylabel('Number of data added to Training')
plt.xlabel('Iteration')
plt.show()
# plot confusion matrix and print metrics
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt="g")
plt.show()
[precision, recall, f1, _] = precision_recall_fscore_support(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print("Accuracy of the model is: ", acc)
print("F1 score of the model is: ", f1)
