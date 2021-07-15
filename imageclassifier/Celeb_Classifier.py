import numpy as np
import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pickle
import csv


celeb_classifiers = {}
def Get_Image_folders(celeb_one,celeb_two):
    file_path = 'First5_all_numpy'
    files = os.listdir(file_path)
    if celeb_one in files:
        celeb_one_data = os.path.join(file_path,celeb_one)
    else:
        celeb_one_data = None
    if celeb_two in files:
        celeb_two_data = os.path.join(file_path, celeb_two)
    else:
        celeb_two_data = None
    if celeb_one_data or celeb_two_data:
        return celeb_one_data, celeb_two_data
    else:
        return None,None



with open('doppelgangers.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        celeb_one = row[0]
        celeb_two = row[1].strip()
        celeb_one_data, celeb_two_data = Get_Image_folders(celeb_one,celeb_two)
        if celeb_one_data != None:
            file_1 = os.listdir(celeb_one_data)
            file_2 = os.listdir(celeb_two_data)
        else:
            continue

        celeb_data = []
        celeb_labels = []
        celeb_one_name = os.path.split(celeb_one_data)[-1]
        celeb_two_name = os.path.split(celeb_two_data)[-1]

        for image_file in file_1:
            image_file_data_path = os.path.join(celeb_one_data, image_file)
            # print(image_file_data_path)
            image_data = np.load(image_file_data_path)
            celeb_data.append(image_data)
            celeb_labels.append(1)

        for image_file2 in file_2:
            image_file_data_path2 = os.path.join(celeb_two_data, image_file2)
            # print(image_file_data_path2)
            image_data = np.load(image_file_data_path2)
            celeb_data.append(image_data)
            celeb_labels.append(-1)

        # These two lines converts the list of arrays into numpy arrays to go through the classifier
        np_celeb_data = np.asarray(celeb_data)
        np_celeb_labels = np.asarray(celeb_labels)
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
                            alpha=1,
                            hidden_layer_sizes=(512, 512),
                            early_stopping=True,
                            # verbose=True
                            )

        print("{}------>{}".format(celeb_one_name, celeb_two_name))
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        if (score - .5)>0:
            print("You look more like {} then {}".format(celeb_one_name,celeb_two_name))
        print(" The mean accuracy of this model is: {}".format(score))
        celeb_classifiers[celeb_one_name, celeb_two_name] = clf

# print(celeb_classifiers)
os.chdir('/home/teacheraccount/Documents/RET2021')
with open("ClassiferPickel", 'wb') as f:
    pickle.dump(celeb_classifiers, f)