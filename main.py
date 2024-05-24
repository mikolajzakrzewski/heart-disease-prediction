# Description: This script trains a random forest classifier on the dataset and evaluates its performance.
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay)

# Prepare training and testing dataframes
X = pd.read_csv('dataset/features.csv', index_col=0)
y = pd.read_csv('dataset/targets.csv', index_col=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.8, random_state=3)

# Create and train the model
clf = RandomForestClassifier(random_state=3)
clf.fit(X_train, y_train.values.ravel())

# Evaluate the model
y_true = y_test.values.ravel()
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0.0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0.0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0.0)

# Display the confusion matrix
stats = f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}'
rcParams.update({'figure.autolayout': True})
disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, cmap='Reds', colorbar=False)
disp.ax_.set_title('Confusion Matrix')
disp.ax_.set_xlabel(disp.ax_.get_xlabel() + '\n\n' + stats)
plt.show()

# Extract feature names and class names
fn = X.columns
cn = y[y.columns[0]].unique()
cn.sort()
cn = cn.astype(str)

# Visualize the first two tree levels
plt.figure(figsize=(10, 5), dpi=200)
tree.plot_tree(clf.estimators_[0], max_depth=2, impurity=False, filled=True,
               feature_names=fn, class_names=cn, rounded=True, fontsize=8)
plt.show()
