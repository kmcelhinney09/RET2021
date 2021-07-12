import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from imutils import paths
import pandas as pd

# set numpy to out put decimals in stead of scientific notation
np.set_printoptions(suppress=True)

'''
    This function calculates the accuracy of the train model using 
    a confession matrix as input
'''


def accuracy(cm):
    diagonal = cm.trace()
    elements = cm.sum()
    return diagonal / elements


# set two empty list to input the numpy arrays from image data to create a set to put through the Classifier
celeb_data = []
celeb_labels = []

# set the name of the target image
target_name = "Beyonce"

# This uses paths from imutils to produce a list of the path of each image in a folder

imagePaths = list(paths.list_images('image_data/cropped_image/Pair_3'))

# print(imagePaths)

'''
    This for loop opens each image in the list of paths above using openCV and saves the numpy array in a single array
    This also extracts the labels from the image path and converts it to an int for the target and non-target

'''

for (i, imagepath) in enumerate(imagePaths):
    # print(i)
    label = imagepath.split(os.path.sep)[-2]

    if label == target_name:
        label = 1
    else:
        label = 0

    image = cv2.imread(imagepath)

    # summarize some details about the image
    image = image / 256
    image = image.reshape(-1)
    image.tolist()
    celeb_labels.append(label)
    celeb_data.append(image)

# These two lines converts the list of arrays into numpy arrays to go through the classifier
np_celeb_data = np.asarray(celeb_data)
np_celeb_labels = np.asarray(celeb_labels)
#print(np.unique(np_celeb_labels))
# print(np_celeb_data.shape)

# Build a train and test split from the given data to put through the classifier
x_train, x_test, y_train, y_test = train_test_split(np_celeb_data, np_celeb_labels, test_size=.2, random_state=1)

print(x_train.shape)
# print(x_train)
# print(y_train.shape)
# print(y_train)
# print(x_test.shape)
# print(x_test)
# print(y_test.shape)
# print(y_test)


# create an MLPClassifier to put imgage data through and fit it
clf = MLPClassifier(solver='sgd',
                    activation='relu',
                    alpha=1e-5,
                    hidden_layer_sizes=(512, 512),
                    n_iter_no_change=10,
                    early_stopping=True,
                    verbose=True
                    )
# Low_loss = 100
# for i in range(100):
#     clf.partial_fit(x_train,y_train,np.unique(celeb_labels))
#     predictions = clf.predict(x_test)
#     model_test_loss = log_loss(y_test,predictions)
#     print("Log loss of test is {}.".format(model_test_loss))
#     print("The Mean Accuracy is {}".format(clf.score(x_test,y_test)))
#     if model_test_loss < Low_loss:
#         Low_loss = model_test_loss
# print("Lowest Loss achieved is {}".format(Low_loss))

clf.fit(x_train,y_train)

# produce a pandas DataFrame from a predict_proba prediction and output a DataFrame with probability the test images
# fit the target label
pd_prediction = pd.DataFrame(clf.predict_proba(x_test), columns=clf.classes_)

# Adds a third column to the DataFrame of the target labels to check the probability
pd_prediction["Labels"] = y_test
print(pd_prediction.round(decimals=5))
print(clf.score(x_test,y_test))

# This section will do a prediction on the data sample and produce a confusion matrix and accuracy for the model
# prediction = clf.predict(x_test)
# acc = confusion_matrix(y_test, prediction)
# print(acc)
# print(accuracy(acc))
