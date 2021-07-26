import csv
import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

'''
The purpose of this script is to take in a series of numpy arrays that are built on the image data of Celebrities that
have been cropped to the face and had their features extracted using InceptionV1 resnet, and train a series of 
MLPClassifiers from sklearn based on doppelganger paris from a .csv
This code was written in a conda environment running Python 3.7.10
Written by Kevin McElhinney with assistance from Nate Thom
7/30/21
'''

# Set up a blank dictionary to save Celebrity name data and trained MLPClassifiers.
celeb_classifiers = {}

'''
This function finds a celebrities folder and builds the path to it.
Input: celeb_one and celeb_two (Names that match the folder name of the celebrity folder, based on 
        doppelganger pairs from .csv
Output: Return the path for each celebrity folder if folder exist returns none if folder does not exists. 
'''


def get_image_folders(celeb_one, celeb_two):
    file_path = 'image_numpy'  # path to folder where all the celebrity folders with numpy arrays are located.
    files = os.listdir(file_path)
    if celeb_one in files:
        celeb_one_data = os.path.join(file_path, celeb_one)
    else:
        celeb_one_data = None
    if celeb_two in files:
        celeb_two_data = os.path.join(file_path, celeb_two)
    else:
        celeb_two_data = None
    if celeb_one_data and celeb_two_data:
        return celeb_one_data, celeb_two_data
    else:
        return None, None


'''
Main portion of code opens the doppelganger csv and reads it in by lines. Using Get_Image_folders() to find the 
path to the celebrity folder. If there is a folder for both celebrities in the pair they are added to a text file if
either have no folder they will be added to a no data file. Numpy's for each celebrity are then grouped together and a
label list is created. The first celebrity in the doppelganger pair is labeled as 1 and the second labeled as 0. 
The data is then used to train a MLPClassifier and the trained model and celebrity names are saved to a dictionary with
the pair of names as a list for the key to the dictionary and the model as the value. The dictionary is then saved as a 
pickle file.
'''


def main():
    with open('doppelgangers_list.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            celeb_one = row[0]
            celeb_two = row[1].strip()
            celeb_one_data, celeb_two_data = get_image_folders(celeb_one, celeb_two)
            if celeb_one_data is not None and celeb_two_data is not None:
                file_1 = os.listdir(celeb_one_data)
                file_2 = os.listdir(celeb_two_data)
                with open("Working_doppelganger_list", "a") as file:  # save list of doppelganger pairs that are trained
                    to_write = celeb_one + ",-------->, " + celeb_two + "\n"
                    file.write(to_write)

            else:
                with open("Pairs_with_no_data", "a") as file:  # save list of pairs that were not trained
                    to_write = celeb_one + ",-------->, " + celeb_two + "\n"
                    file.write(to_write)
                continue

            # setting up empty lists to join celeb_one and celeb_two numpy array data and to create labels
            celeb_data = []
            celeb_labels = []

            # extract celeb names based on folder names
            celeb_one_name = os.path.split(celeb_one_data)[-1]
            celeb_two_name = os.path.split(celeb_two_data)[-1]

            # loop over numpy arrays in celeb_one folder, add data to celeb_data list place 1 in celeb_label for each
            for image_file in file_1:
                image_file_data_path = os.path.join(celeb_one_data, image_file)
                image_data = np.load(image_file_data_path)
                celeb_data.append(image_data)
                celeb_labels.append(1)

            # loop over numpy arrays in celeb_two folder, add data to celeb_data list place 0 in celeb_label for each
            for image_file2 in file_2:
                image_file_data_path2 = os.path.join(celeb_two_data, image_file2)
                image_data = np.load(image_file_data_path2)
                celeb_data.append(image_data)
                celeb_labels.append(-1)

            # These two lines converts the list of arrays into numpy arrays to go through the classifier
            np_celeb_data = np.asarray(celeb_data)
            np_celeb_labels = np.asarray(celeb_labels)

            # Build a train and test split from the given data to put through the classifier taking 20% to test
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
                                # verbose=True
                                )

            print("{}------>{}".format(celeb_one_name, celeb_two_name))  # provide feedback for progress

            clf.fit(x_train, y_train)

            # Find accuracy for output file for later review
            score = clf.score(x_test, y_test)

            with open("Accuracy Output", 'a') as file:  # Creating output file for accuracy data
                write1 = "{}------>{} \n".format(celeb_one_name, celeb_two_name)
                write2 = " The mean accuracy of this model is: {} \n".format(score)
                file.write(write1)
                file.write(write2)

            # Update celeb_classifiers dictionary with current celebrity names and the classifier trained on their data
            celeb_classifiers[celeb_one_name, celeb_two_name] = clf


main()

# Save celeb_classifiers dictionary as a pickle to input into comparison scripts.
with open("Saved_Classifiers.pkl", 'wb') as f:
    pickle.dump(celeb_classifiers, f)

