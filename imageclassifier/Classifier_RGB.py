import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from imutils import paths
import pandas as pd

# set numpy to out put deciamals in stead of scientific notation
np.set_printoptions(suppress=True)

'''
    This funciton calculates the accuracy of the train model using 
    a confussion matrix as input
'''
def accuracy(confusionMatrix):
    diagonal = confusionMatrix.trace()
    elements = confusionMatrix.sum()
    return diagonal/elements


# set two empty list to input the numpy arrarys from image data to create a set to put through the Classifier
celeb_data = []
celeb_labels = []


# set the name of the target image
target_name = "Zooey_Deschanel"

# This uses paths from imutils to produce a list of the path of each image in a folder
imagePaths = list(paths.list_images('image_data/cropped_image/Part_2'))

'''
    This for loop opens each image in the list of paths above using openCV and saves the numpy array in a single array
    This also extracts the lables from the image path and converts it to an int for the target and non-target

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

#These two lines converts the list of arrays into numpy arrays to go through the classifier
np_celeb_data = np.asarray(celeb_data)
np_celeb_labels = np.asarray(celeb_labels)


#Build a train and test split from the given data to put through the classifier
x_train, x_test, y_train, y_test = train_test_split(np_celeb_data,np_celeb_labels,test_size=.2,random_state=1)
'''
print(x_train.shape)
print(x_train)
print(y_train.shape)
print(y_train)
print(x_test.shape)
print(x_test)
print(y_test.shape)
print(y_test)
'''

#create an MLPClassifier to put imgage data through and fit it
clf = MLPClassifier(solver='lbfgs',activation='relu', alpha = 1e-5, hidden_layer_sizes=(128,128))
clf.fit(x_train,y_train)

#produce a pandas DataFrame from a predict_proba prediction and output a DataFrame with probaility the test images
#fit the target label
pd_prediction = pd.DataFrame(clf.predict_proba(x_test), columns=clf.classes_)

# Adds a third colmn to the DataFrame of the target labels to check the probability
pd_prediction["Labels"] = y_test
print(pd_prediction.round(decimals=5))

#This section will do a prediction on the data sample and produce a confusion matrix and accuracy for the model
prediction = clf.predict(x_test)
acc = confusion_matrix(y_test,prediction)
print(acc)
print(accuracy(acc))


