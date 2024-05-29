# Description: This script evaluates the performance of the trained model
import os
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import rcParams
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

# Load the model data
with open('model/classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('model/train_data.pkl', 'rb') as f:
    X_train = pickle.load(f)
    y_train = pickle.load(f)

with open('model/test_data.pkl', 'rb') as f:
    X_test = pickle.load(f)
    y_test = pickle.load(f)

with open('model/feature_data.pkl', 'rb') as f:
    fn = pickle.load(f)
    cn = pickle.load(f)
    dl = pickle.load(f)

# Make a directory to store model evaluation reports
if not os.path.exists('evaluation'):
    os.makedirs('evaluation', exist_ok=True)

# Generate a classification report
y_true = y_test.values.ravel()
y_pred = clf.predict(X_test)
report = classification_report(y_true, y_pred, target_names=dl, zero_division=0.0, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('evaluation/classification_report.csv')

# Save the confusion matrix to a file
rcParams.update({'figure.autolayout': True})
sns.set_context('talk')
disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=dl, cmap='Reds', colorbar=False)
disp.ax_.set_title('Confusion Matrix')
plt.savefig('evaluation/confusion_matrix.png')
plt.clf()

# Save feature importance plots to files
plt.figure(figsize=(6, 6))

fi = pd.Series(clf.feature_importances_, index=fn)
fi = fi.sort_values(ascending=False)
sns.reset_defaults()
sns.set_style('darkgrid', {'patch.edgecolor': "black", 'patch.linewidth': 0.5})
sns.set_context('talk')
plot = sns.barplot(x=fi.values, y=fi.index, hue=fi.index, palette='Reds_r', color='red')
plot.set_title('Impurity-based feature importance')
plot.set_xlabel('Mean impurity decrease')
plot.set_ylabel('Feature')
plt.savefig('evaluation/impurity_feature_importance.png')
plt.clf()

pi = permutation_importance(clf, X_test, y_test, n_repeats=10, scoring='f1_weighted')
fi = pd.Series(pi.importances_mean, index=fn)
fi = fi.sort_values(ascending=False)
sns.reset_defaults()
sns.set_style('darkgrid', {'patch.edgecolor': "black", 'patch.linewidth': 0.5})
sns.set_context('talk')
plot = sns.barplot(x=fi.values, y=fi.index, hue=fi.index, palette='Reds_r')
plot.set_title('Permutation-based feature importance')
plot.set_xlabel('Mean accuracy decrease')
plot.set_ylabel('Feature')
plt.savefig('evaluation/permutation_feature_importance.png')
plt.clf()
