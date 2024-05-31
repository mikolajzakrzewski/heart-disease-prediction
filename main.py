# Description: This script trains a random forest classifier on the dataset and tunes its hyperparameters
import os
import pickle

import pandas as pd

from sklearn import tree
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Prepare the dataset in the form of dataframes
X = pd.read_csv('dataset/features.csv', index_col=0)
y = pd.read_csv('dataset/targets.csv', index_col=0)

# Deal with missing values
X = X.fillna(X.mean())

# Omit the severity of the disease by replacing target values 2, 3, and 4 with 1
y = y.replace([2, 3, 4], 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.8)

# Create and train the model
clf = RandomForestClassifier(class_weight='balanced_subsample', n_jobs=-1)
clf.fit(X_train, y_train.values.ravel())

# Tune the hyperparameters
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
if not os.path.exists('model'):
    os.makedirs('model', exist_ok=True)

# Save tuning results to a file
cv_results_df = pd.DataFrame(clf_tuned.cv_results_)
cv_results_df.to_csv('model/cv_results.csv')

# Extract feature names, class names and class labels
fn = X.columns
cn = y[y.columns[0]].unique()
cn.sort()
cn = cn.astype(str)
dl = ['Absence', 'Presence']

# Save trees' text representations to files
if os.path.exists('model/trees'):
    for file in os.listdir('model/trees'):
        os.remove(f'model/trees/{file}')

else:
    os.makedirs('model/trees', exist_ok=True)

for i, estimator in enumerate(clf.estimators_):
    with open(f'model/trees/tree_{i + 1}', 'w') as f:
        f.write(tree.export_text(estimator, feature_names=fn, class_names=dl))

# Serialize model info to files
with open('model/classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('model/train_data.pkl', 'wb') as f:
    pickle.dump(X_train, f)
    pickle.dump(y_train, f)

with open('model/test_data.pkl', 'wb') as f:
    pickle.dump(X_test, f)
    pickle.dump(y_test, f)

with open('model/feature_data.pkl', 'wb') as f:
    pickle.dump(fn, f)
    pickle.dump(cn, f)
    pickle.dump(dl, f)
