# Description: This script trains a random forest classifier on the dataset and tunes its hyperparameters
import sys
import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn import tree
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report,
                             ConfusionMatrixDisplay)

# Prepare training and testing dataframes
X = pd.read_csv('dataset/features.csv', index_col=0)
y = pd.read_csv('dataset/targets.csv', index_col=0)

# Deal with missing values
X = X.fillna(X.mean())

# Ask the user if they want to omit the severity of the disease
if len(sys.argv) > 1:
    omit_severity = sys.argv[1]
else:
    omit_severity = input('Omit the severity of the disease? (y/n): ')
if omit_severity == 'y':
    # Omit the severity of the disease by replacing target values 2, 3, and 4 with 1
    y = y.replace([2, 3, 4], 1)
    dl = ['Absence', 'Presence']
else:
    # Retain the severity of the disease
    dl = ['Absence', 'Mild', 'Moderate', 'Severe', 'Critical']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.8)

# Create and train the model
clf = RandomForestClassifier(class_weight='balanced_subsample', n_jobs=-1)
clf.fit(X_train, y_train.values.ravel())

# Hyperparameter tuning
params = {'n_estimators': [10, 50, 100, 200, 500],
          'max_depth': [None, 10, 20, 50, 100],
          'min_samples_split': [2, 5, 10, 20],
          'min_samples_leaf': [1, 2, 5, 10],
          'max_features': ['sqrt', 'log2', None],
          'max_leaf_nodes': [None, 10, 20, 50, 100]}
clf_tuned = RandomizedSearchCV(clf, params, n_jobs=-1, scoring='f1_weighted')
clf_tuned.fit(X_train, y_train.values.ravel())
clf.set_params(**clf_tuned.best_params_)
clf.fit(X_train, y_train.values.ravel())

# Make a directory to store various reports and statistics
if not os.path.exists('reports'):
    os.mkdir('reports')

# Save tuning results to a file
cv_results_df = pd.DataFrame(clf_tuned.cv_results_)
cv_results_df.to_csv('reports/cv_results.csv')

# Evaluate the model
y_true = y_test.values.ravel()
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0.0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0.0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0.0)
report = classification_report(y_true, y_pred, target_names=dl, zero_division=0.0, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('reports/classification_report.csv')

# Save the confusion matrix to a file
stats = f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}'
rcParams.update({'figure.autolayout': True})
disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=dl, cmap='Reds', colorbar=False)
disp.ax_.set_title('Confusion Matrix')
disp.ax_.set_xlabel(disp.ax_.get_xlabel() + '\n\n' + stats)
plt.savefig('reports/confusion_matrix.png')

# Extract feature names and class names
fn = X.columns
cn = y[y.columns[0]].unique()
cn.sort()
cn = cn.astype(str)

# Save trees' text representations to files
if os.path.exists('reports/trees'):
    for file in os.listdir('reports/trees'):
        os.remove(f'reports/trees/{file}')

else:
    os.mkdir('reports/trees')

for i, estimator in enumerate(clf.estimators_):
    with open(f'reports/trees/tree_{i + 1}', 'w') as f:
        f.write(tree.export_text(estimator, feature_names=fn, class_names=dl))

# Save various generated data to files for further work
with open('reports/model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('reports/train_data.pkl', 'wb') as f:
    pickle.dump(X_train, f)
    pickle.dump(y_train, f)

with open('reports/test_data.pkl', 'wb') as f:
    pickle.dump(X_test, f)
    pickle.dump(y_test, f)

with open('reports/feature_data.pkl', 'wb') as f:
    pickle.dump(fn, f)
    pickle.dump(cn, f)
    pickle.dump(dl, f)
