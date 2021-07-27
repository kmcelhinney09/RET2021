
import pandas as pd
import os
import pickle as pkl
import numpy as np
from tqdm import tqdm

# load classifiers from pickle saved from Doppelganger_Classifier
pickle_file = open("Saved_Classifiers.pkl", 'rb')
classifer_dict = pkl.load(pickle_file)
pickle_file.close()

#file path to celebA numpys created using facenet and their associated labels document
celebA_labels_path = 'identity_CelebA.txt'
celebA_numpies_path = 'img_align_celeba_numpy'


celeba_numpies = os.listdir(celebA_numpies_path)
celebA_predictions = []
celebA_predictions_labels = []


with open(celebA_labels_path, 'r') as labels_file:
    label_lines = labels_file.readlines()
    for line in tqdm(label_lines):
        celeba_line = line.split()
        file = celeba_line[0]
        label = celeba_line[1]
        numpy_file_name = file + ".npy"
        if numpy_file_name in celeba_numpies:
            print(numpy_file_name)
            image_match_probability = []
            numpy_path = os.path.join(celebA_numpies_path,numpy_file_name)
            celebA_numpy = np.load(numpy_path)
            # print(celebA_numpy.shape)
            celebA_numpy = celebA_numpy.reshape(1,-1)
            # print(celebA_numpy.shape)

            for name, each_classifier in classifer_dict.items():
                probability = each_classifier.predict_proba(celebA_numpy)
                image_match_probability.append(probability[0][1])
                print(name, len(image_match_probability))
                celebA_predictions.append(image_match_probability)
                celebA_predictions_labels.append(label)
try:
    celebA_predictions_numpy = np.asarray(celebA_predictions)
    celebA_predictions_labels_numpy = np.asarray(celebA_predictions_labels)

    print(celebA_predictions_numpy.shape)
    print(celebA_predictions_labels_numpy.shape)

    with open('CelebA_prediction_numpy', 'wb') as cApn:
        pkl.dump(celebA_predictions_numpy,cApn)

    with open('CelebA_predictions_numpy_labels', 'wb') as cApnL:
        pkl.dump(celebA_predictions_labels_numpy, cApnL)

    with open('CelebA_prediction', 'wb') as cAp:
        pkl.dump(celebA_predictions, cAp)

    with open('CelebA_prediction_labels','wb') as cApL:
        pkl.dump(celebA_predictions_numpy,cApL)




