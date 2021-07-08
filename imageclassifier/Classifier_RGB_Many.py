import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from imutils import paths
import pandas as pd
import contextlib

# This will out put the console data to a file for review
with open('Classifier_Output.txt', 'w') as f:
    with contextlib.redirect_stdout(f):

        # set numpy to out put decimals in stead of scientific notation
        np.set_printoptions(suppress=True)

        '''
            This function calculates the accuracy of the train model using 
            a confession matrix as input
        '''

        # set two empty list to input the numpy arrays from image data to create a set to put through the Classifier
        celeb_data = []
        celeb_labels = []

        # This uses paths from imutils to produce a list of the path of each image in a folder
        Paths = os.listdir("image_data/cropped_image")
        for each in range(len(Paths)):

            pairing_path = os.path.join("image_data/cropped_image", Paths[i])

            imagePaths = list(paths.list_images(pairing_path))
            # set the name of the target image
            celb_list = os.listdir(pairing_path)
            target_name = celb_list[0]

            '''
                This for loop opens each image in the list of paths above using openCV and saves the numpy array in a single 
                array
                This also extracts the labels from the image path and converts it to an int for the target and non-target
            
            '''

            for (index, image_path) in enumerate(imagePaths):
                # print(i)
                label = image_path.split(os.path.sep)[-2]

                if label == target_name:
                    label = 1
                else:
                    label = 0

                image = cv2.imread(image_path)

                # summarize some details about the image
                image = image / 256
                image = image.reshape(-1)
                image.tolist()
                celeb_labels.append(label)
                celeb_data.append(image)

            # These two lines converts the list of arrays into numpy arrays to go through the classifier
            np_celeb_data = np.asarray(celeb_data)
            np_celeb_labels = np.asarray(celeb_labels)
            # print(np.unique(np_celeb_labels))
            # print(np_celeb_data.shape)

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
                                alpha=1e-5,
                                hidden_layer_sizes=(512, 512),
                                n_iter_no_change=20,
                                early_stopping=True,
                                verbose=True
                                )

            print("{}------>{}".format(celb_list[0], celb_list[1]))
            clf.fit(x_train, y_train)

            # produce a pandas DataFrame from a predict_proba prediction and output a DataFrame with probability the
            # test images fit the target label
            pd_prediction = pd.DataFrame(clf.predict_proba(x_test), columns=clf.classes_)

            # Adds a third column to the DataFrame of the target labels to check the probability
            pd_prediction["Labels"] = y_test
            print(pd_prediction.round(decimals=5))
            # print the accuracy score of the model to the console
            print(clf.score(x_test, y_test))
