import numpy as np
import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

pair = "First5_numpy/Pair_5"
os.chdir(pair)
image_dir = os.listdir()
file_1 = os.listdir(image_dir[0])
file_2 = os.listdir(image_dir[1])
celeb_data = []
celeb_labels = []

for image_file in file_1:
    image_file_data_path = os.path.join(image_dir[0], image_file)
    image_data = np.load(image_file_data_path)
    celeb_data.append(image_data)
    celeb_labels.append(1)

for image_file2 in file_2:
    image_file_data_path = os.path.join(image_dir[1], image_file2)
    image_data = np.load(image_file_data_path)
    celeb_data.append(image_data)
    celeb_labels.append(0)

# These two lines converts the list of arrays into numpy arrays to go through the classifier
np_celeb_data = np.asarray(celeb_data)
np_celeb_labels = np.asarray(celeb_labels)
print(np_celeb_data.shape)
# Build a train and test split from the given data to put through the classifier
x_train, x_test, y_train, y_test = train_test_split(
    np_celeb_data,
    np_celeb_labels,
    test_size=.2,
    random_state=1
)

# create an MLPClassifier to put image data through and fit it
clf = MLPClassifier(solver='sgd',
                    activation='relu',
                    alpha=1,
                    hidden_layer_sizes=(512, 512),
                    early_stopping=True,
                    verbose=True
                    )

print("{}------>{}".format(image_dir[0], image_dir[1]))
clf.fit(x_train, y_train)

# produce a pandas DataFrame from a predict_proba prediction and output a DataFrame with probability the
# test images fit the target label
pd_prediction = pd.DataFrame(clf.predict_proba(x_test), columns=clf.classes_)

# Adds a third column to the DataFrame of the target labels to check the probability
pd_prediction["Labels"] = y_test
print(pd_prediction.round(decimals=5))
# print the accuracy score of the model to the console
print(" The mean accuracy of this model is: {}".format(clf.score(x_test, y_test)))