# running might take upto 1 minute please be patient
import pandas as pd
import seaborn as sn  # for plotting purposes
import matplotlib.pyplot as plt
import numpy as np
from random import sample
from sklearn.utils import shuffle       # for shuffling dataset

df = pd.read_csv("prison_dataset.csv")  # load data
# shuffle dataset
df=shuffle(df,random_state=0)
training_data = df.sample(frac=0.8, random_state=0)
testing_data = df.drop(training_data.index)

attributes = list(df.columns)
print(attributes)
attributes.remove("Recidivism - Return to Prison numeric")


class node:
    def __init__(self):
        self.sub_nodes = []
        self.isLeaf = False
        self.attributes = ""
        self.result = ""


# calculates entropy of the given node
def calculate_entropy(node):
    dups = node.pivot_table(index=['Recidivism - Return to Prison numeric'], aggfunc='size')
    try:
        negative = dups.loc[0]
    except KeyError:
        negative = 0
    try:
        positive = dups.loc[1]
    except KeyError:
        positive = 0
    p = positive / (positive + negative)
    n = negative / (positive + negative)
    if p == 0 or n == 0:
        entropy = 0
    else:
        entropy = -p * np.log2(p) - n * np.log2(n)
    return entropy


# calculates information gain of the given node
def info_gain(data, attr):
    labels = []
    [labels.append(x) for x in data[attr] if x not in labels]
    root_entropy = calculate_entropy(data)
    node_entropy = []
    node_num_of_values = []
    for label in labels:
        leaf = data[data[attr] == label]
        ent = calculate_entropy(leaf)
        node_entropy.append(ent)
        node_num_of_values.append(len(leaf))
    gain = root_entropy
    for i in range(len(labels)):
        gain = gain - (float(node_num_of_values[i]) / float(len(data))) * node_entropy[i]
    return gain


# uses ID3 algorithm to make the tree
def make_tree(data, attrs, depth):
    root = node()
    gain = []
    feature_ls = list()
    max_features = round(np.sqrt(len(attrs)))
    feature_ls = sample(attrs, max_features)
    for attr in feature_ls:
        gain.append(info_gain(data, attr))
        if max(gain) == gain[-1]:
            best_attr = attr
    root.attributes = best_attr
    labels = []
    [labels.append(x) for x in data[best_attr] if x not in labels]
    for label in labels:
        node_data = data[data[best_attr] == label].reset_index(drop=True)
        if calculate_entropy(node_data) != 0 and depth > 0:  # if the node is not leaf
            curr_node = node()
            curr_node.attributes = label
            new_attrs = attrs.copy()
            new_attrs.remove(best_attr)
            next_node = make_tree(node_data, new_attrs, depth - 1)
            curr_node.sub_nodes.append(next_node)
            root.sub_nodes.append(curr_node)
        else:  # if the node is leaf
            leaf_node = node()
            leaf_node.isLeaf = True
            leaf_node.attributes = label
            leaf_node.result = node_data["Recidivism - Return to Prison numeric"][0]
            root.sub_nodes.append(leaf_node)
    return root


# function to predict Recidivism for each row in test dataframe
def predict(row, tree):
    result = node()
    attributes = list(row.columns)
    for attr in attributes:
        if attr == tree.attributes:
            for subs in tree.sub_nodes:
                if row[attr].item() == subs.attributes:
                    result = tree.sub_nodes[tree.sub_nodes.index(subs)]
                    if len(result.sub_nodes) == 1:
                        result = result.sub_nodes[0]

            if len(result.sub_nodes) != 0:
                return predict(row, result)
            else:
                return result.result


# takes a bootstrap sample to train random forest
def bootstrap_sample(X_train, y_train):
    X_bootstrap = X_train.sample(frac=0.7, replace=True)
    y_bootstrap = y_train[X_bootstrap.index]
    return X_bootstrap, y_bootstrap


# random forest algorithm, n_estimators is the number of trees
def random_forest(X_train, y_train, attrs, n_estimators, max_depth):
    trees = list()
    for i in range(n_estimators):
        X_bootstrap, y_bootstrap = bootstrap_sample(X_train, y_train)
        tree = make_tree(X_bootstrap, attrs, max_depth)
        trees.append(tree)
    return trees


# function to predict the result based on the trained random forest
def maj_predict(trees, X_test):
    predicted = []
    for i in range(len(X_test.index)):
        dff = X_test.iloc[[i]].drop("Recidivism - Return to Prison numeric", axis=1)
        pred = []
        for tree in trees:
            pred.append(predict(dff, tree))
        maj_pred = max(pred, key=pred.count)
        predicted.append(maj_pred)

    return (predicted)


y_train = training_data["Recidivism - Return to Prison numeric"]
# you can change depth and number of trees
trees = random_forest(training_data, y_train, attributes, n_estimators=100, max_depth=3)
y_pred = maj_predict(trees, testing_data)

actu = testing_data["Recidivism - Return to Prison numeric"].tolist()
data = {'y_Actual': actu,
        'y_Predicted': y_pred
        }
df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
# calculate confusion matrix and accuracy
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print('\n', confusion_matrix)
accuracy = (confusion_matrix[1][1] + confusion_matrix[0][0]) / (
        confusion_matrix[1].sum() + confusion_matrix[0].sum())

print("Accuracy of prediction is equal to: ", accuracy * 100, "%")
sn.heatmap(confusion_matrix, annot=True, cmap="icefire", linewidths=1, fmt='g')
plt.show()
