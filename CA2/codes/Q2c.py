import pandas as pd
import numpy as np
import seaborn as sn  # for plotting purposes
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.utils import shuffle       # for shuffling dataset

# encoder to encode string labels to integer values
label_encoder = preprocessing.LabelEncoder()
clf = RandomForestClassifier(criterion='entropy', max_depth=3, n_estimators=100, random_state=0)

df = pd.read_csv("prison_dataset.csv")  # load data
# shuffle dataset
df=shuffle(df,random_state=0)
dff = df.copy()
for feat in df.columns:  # encode string labels to numbers
    label_encoder.fit(df[feat])
    dff[feat] = label_encoder.transform(df[feat])

# split train and test data
labels = dff["Recidivism - Return to Prison numeric"]
features = dff.drop("Recidivism - Return to Prison numeric", axis=1)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,
                                                                            random_state=0)
# train th model
clf.fit(train_features, train_labels)
# predict the test data
y_pred = clf.predict(test_features)
# measure accuracy and make confusion matrix
confusion_matrix = pd.crosstab(y_pred, test_labels, rownames=['Predicted'], colnames=['Actual'])
print(confusion_matrix)
print("Accuracy: ", metrics.accuracy_score(test_labels, y_pred) * 100)
sn.heatmap(confusion_matrix, annot=True, fmt='g', cmap="viridis",
           xticklabels=["non-Recidivist", "Recidivist"], yticklabels=["non-Recidivist", "Recidivist"])
plt.show()
