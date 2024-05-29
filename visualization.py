# Description: This script generates visualizations of one of the model's trees and saves them to files
import os
import pickle
import sys

import graphviz
import dtreeviz
from sklearn import tree

with open('model/classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('model/train_data.pkl', 'rb') as f:
    X_train = pickle.load(f)
    y_train = pickle.load(f)

with open('model/feature_data.pkl', 'rb') as f:
    fn = pickle.load(f)
    _ = pickle.load(f)
    dl = pickle.load(f)

if not os.path.exists('visualization/dtreeviz'):
    os.makedirs('visualization/dtreeviz', exist_ok=True)

if not os.path.exists('visualization/graphviz'):
    os.makedirs('visualization/graphviz', exist_ok=True)

if len(sys.argv) > 2:
    tree_num = int(sys.argv[1])
    mode = sys.argv[2]
else:
    tree_num = int(input('Enter the number of the tree to visualize: '))
    mode = input('Enter the mode of visualization (full or partial): ')

if mode == 'partial':
    if len(sys.argv) > 3:
        levels_num = int(sys.argv[3])
    else:
        levels_num = int(input('Enter the number of levels to display: '))
    dot_file = tree.export_graphviz(clf.estimators_[tree_num], max_depth=levels_num, feature_names=fn, class_names=dl,
                                    filled=True, rounded=True)
    graphviz.Source(dot_file).render('visualization/graphviz/partial_tree', format='png', cleanup=True)

    viz = dtreeviz.trees.model(clf.estimators_[tree_num], X_train, y_train.values.ravel(),
                               target_name='Heart disease presence',
                               feature_names=fn, class_names=dl)
    viz.view(depth_range_to_display=(0, levels_num), scale=1.3).save('visualization/dtreeviz/partial_tree.svg')
    os.remove('visualization/dtreeviz/partial_tree')

elif mode == 'full':
    dot_file = tree.export_graphviz(clf.estimators_[tree_num], feature_names=fn, class_names=dl, filled=True,
                                    rounded=True)
    graphviz.Source(dot_file).render('visualization/graphviz/full_tree', format='png', cleanup=True)

    viz = dtreeviz.trees.model(clf.estimators_[tree_num], X_train, y_train.values.ravel(),
                               target_name='Heart disease presence',
                               feature_names=fn, class_names=dl)
    viz.view().save('visualization/dtreeviz/full_tree.svg')
    os.remove('visualization/dtreeviz/full_tree')
