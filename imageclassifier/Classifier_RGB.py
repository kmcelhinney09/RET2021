import os
import numpy
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from imutils import paths
import pprint

# sample.png is the name of the image
# file and assuming that it is uploaded
# in the current directory or we need
# to give the path
image_paths = list(paths.list_images('image_data/cropped_image'))
#print(image_paths)
celeb_data = []
celeb_labels = []
for i,imagepath in enumerate(image_paths):
    #print(i)
    label = imagepath.split(os.path.sep)[-2]
    img_file_name = imagepath.split(os.path.sep)[-2]
    img_full_path = image_paths[i]
    image = Image.open(img_full_path)

    if label == "Brad_Pit":
        label = 1
    else:
        label = 0

    # summarize some details about the image
    numpydata = numpy.asarray(image)
    numpydata = numpydata/256
    numpydata = numpydata.reshape(-1)


    print(numpydata.shape)
    pprint.pprint(numpydata)

    celeb_labels.append(label)
    celeb_data.append(numpydata)


x_train, x_test, y_train, y_test = train_test_split(celeb_data,celeb_labels,test_size=.2,random_state=1)

clf = MLPClassifier(solver='adam',activation='relu', hidden_layer_sizes=(64,64))

clf.fit(x_train,y_train)



