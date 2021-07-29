import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
celebA_data_path = 'CelebA_prediction_numpy'
celebA_labels_path = 'CelebA_predictions_numpy_labels'

with open(celebA_data_path, 'rb') as data:
    celebA_data = pickle.load(data)

with open(celebA_labels_path, 'rb') as labels:
    celebA_labels = pickle.load(labels)

print(celebA_data.shape)
print(celebA_labels.shape)

x_train, x_test, y_train, y_test = train_test_split(
    celebA_data,
    celebA_labels,
    test_size=.2,
    random_state=1
)
# create an MLPClassifier to put image data through and fit it
clf = Perceptron(tol=1e-3, random_state=0, verbose=True)

# clf = DecisionTreeClassifier(random_state=0, max_depth=250)

# clf = LinearSVC(random_state=0,tol=1e-5, verbose=True)

clf.fit(x_train, y_train, )

# Find accuracy for output file for later review
score = clf.score(x_test, y_test)
print(score)

# depth = clf.get_depth()
# print(depth)
#
# num_leaves = clf.get_n_leaves()
# print(num_leaves)

