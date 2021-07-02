import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from imutils import paths
import pandas as pd

np.set_printoptions(suppress=True)

def accuracy(confusionMatrix):
    diagonal = confusionMatrix.trace()
    elements = confusionMatrix.sum()
    return diagonal/elements


#image_paths = list(paths.list_images('image_data/cropped_image'))
#print(image_paths)
celeb_data = []
celeb_labels = []
imagePaths = list(paths.list_images('image_data/cropped_image'))
#print(imagePaths)
for (i, imagepath) in enumerate(imagePaths):
    # print(i)
    label = imagepath.split(os.path.sep)[-2]

    if label == "Brad_Pit":
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

np_celeb_data = np.asarray(celeb_data)
np_celeb_labels = np.asarray(celeb_labels)

#print(celeb_labels)
print(np_celeb_data.shape)


x_train, x_test, y_train, y_test = train_test_split(np_celeb_data,np_celeb_labels,test_size=.2,random_state=1)
#print(x_train.shape)
#print(y_train.shape)
clf = MLPClassifier(solver='lbfgs',activation='relu', alpha = 1e-5, hidden_layer_sizes=(128,128))

clf.fit(x_train,y_train)


pd_prediction = pd.DataFrame(clf.predict_proba(x_test), columns=clf.classes_)
pd_prediction["Labels"] = y_test
print(pd_prediction.round(decimals=5))




