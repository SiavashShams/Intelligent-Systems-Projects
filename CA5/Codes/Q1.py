import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("Mushroom_Train.csv")
df_test = pd.read_csv("Mushroom_Test.csv")

ss = df_train["class"].value_counts()
# prior probabilities
prior_e = ss.loc["e"] / len(df_train["class"])
prior_p = ss.loc["p"] / len(df_train["class"])

X_train_e = df_train[df_train["class"] == 'e']
X_train_p = df_train[df_train["class"] == 'p']
X_train_e = X_train_e.drop("class", axis=1)
X_train_p = X_train_p.drop("class", axis=1)
# likelihoods
probs_e = {}
probs_p = {}
for [colname, col] in X_train_e.iteritems():
    probs_e[colname] = col.value_counts(normalize=True)
for [colname, col] in X_train_p.iteritems():
    probs_p[colname] = col.value_counts(normalize=True)

# predict label for test data
def predict(x):
    # calculate posterior probabilities
    posterior_e = prior_e
    posterior_p = prior_p
    for col, val in x.iteritems():
        try:
            posterior_e *= probs_e[col][val]
        except KeyError:
            posterior_e *= 0
        try:
            posterior_p *= probs_p[col][val]
        except KeyError:
            posterior_p *= 0
    # classify
    if posterior_e > posterior_p:
        label = 'e'
    else:
        label = 'p'
    return label


# Test data
predicted = []

for i in df_test.iterrows():
    predicted.append(predict(i[1].drop("class")))
predicted = pd.DataFrame(predicted)
confusion_matrix = pd.crosstab(predicted[0], df_test['class'], rownames=['Actual'], colnames=['Predicted'])
print("Accuracy:", (np.array(confusion_matrix)[0][0]+np.array(confusion_matrix)[1][1]) / len(df_test))
print(confusion_matrix)
sns.heatmap(confusion_matrix, annot=True, fmt="g", cmap="Blues")
plt.show()
