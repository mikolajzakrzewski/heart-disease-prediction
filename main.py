# Description: This script trains a random forest classifier on the dataset and evaluates its performance.
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn import tree
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay)

# Prepare training and testing dataframes
X = pd.read_csv('dataset/features.csv', index_col=0)
y = pd.read_csv('dataset/targets.csv', index_col=0)

# Deal with missing values
X = X.fillna(X.mean())

# Ask the user if they want to omit the severity of the disease
omit_severity = input('Omit the severity of the disease? (y/_): ')
if omit_severity == 'y':
    # Omit the severity of the disease by replacing target values 2, 3, and 4 with 1
    y = y.replace([2, 3, 4], 1)
    dl = ['Absence', 'Presence']
else:
    # Retain the severity of the disease
    dl = ['Absence', 'Mild', 'Moderate', 'Severe', 'Critical']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.8, random_state=3)

# Create and train the model
clf = RandomForestClassifier(n_jobs=-1, random_state=3)
clf.fit(X_train, y_train.values.ravel())

# Hyperparameter tuning
# TODO: Adjust the hyperparameter search space
params = {'n_estimators': [10, 50, 100, 200, 500],
          'criterion': ['gini', 'entropy', 'log_loss'],
          'max_depth': [None, 10, 20, 50, 100],
          'min_samples_split': [2, 5, 10],
          'min_samples_leaf': [1, 2, 4],
          'max_features': ['sqrt', 'log2', None],
          'max_leaf_nodes': [None, 10, 20, 50, 100],
          'bootstrap': [True, False],
          'class_weight': ['balanced', 'balanced_subsample', None]}
clf_tuned = RandomizedSearchCV(clf, params, n_jobs=-1, random_state=3)
clf_tuned.fit(X_train, y_train.values.ravel())
clf.set_params(**clf_tuned.best_params_)
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
disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=dl, cmap='Reds', colorbar=False)
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
tree.plot_tree(clf.estimators_[0], max_depth=2, filled=True, feature_names=fn, class_names=cn,
               rounded=True, fontsize=8)
plt.show()
