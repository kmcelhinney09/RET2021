import os
import pickle as pkl
import numpy as np
from tqdm import tqdm

# load classifiers from pickle saved from Doppelganger_Classifier
pickle_file = open("Saved_Classifiers.pkl", 'rb')
classifier_dict = pkl.load(pickle_file)
pickle_file.close()

# file path to celebA numpys created using facenet and their associated labels document
celebA_labels_path = 'identity_CelebA.txt'
celebA_numpies_path = 'img_align_celeba_numpy'

celebA_sample_size = 20000  # change this number to slice out different portions of the over 200,000 celbA numpys

# create a list of a slice celebA numbys to iterate over and bring in
celeba_numpies = os.listdir(celebA_numpies_path)[:celebA_sample_size]

# empty lists for the resulting predictions made using the doppelganger classifiers and associated labels
celebA_predictions = []
celebA_predictions_labels = []

'''
This section of code will open the celebA labels document create a list from a same slice as the celebA numpys
Import the numpys of that label, run it through the doppelganger classifiers and append the results to the 
celebA_predictions list, and the file label to the celebA_predictions_labels list
'''
with open(celebA_labels_path, 'r') as labels_file:
    lines_read = labels_file.readlines()
    label_lines = lines_read[:celebA_sample_size]

    # this will iterate over each filename/label pair split it and loads the numpy of that file name
    for line in tqdm(label_lines):
        celeba_line = line.split()
        file = celeba_line[0]
        label = celeba_line[1]
        numpy_file_name = file + ".npy"
        if numpy_file_name in celeba_numpies:  # double check that the file we want to load is actually in the directory
            # print(numpy_file_name)
            image_match_probability = []
            numpy_path = os.path.join(celebA_numpies_path, numpy_file_name)
            celebA_numpy = np.load(numpy_path)
            # print(celebA_numpy.shape)
            celebA_numpy = celebA_numpy.reshape(1, -1)
            # print(celebA_numpy.shape)

            # this is the part that runs the numpy from CelebA through the doppelganger classifiers
            for name, each_classifier in classifier_dict.items():
                probability = each_classifier.predict_proba(celebA_numpy)
                image_match_probability.append(probability[0][1])
                # print(name, len(image_match_probability))
                celebA_predictions.append(image_match_probability)
                celebA_predictions_labels.append(label)

# Here we will try to convert our lists of predictions into numpy arrays and then pickle them
try:
    celebA_predictions_numpy = np.asarray(celebA_predictions)
    celebA_predictions_labels_numpy = np.asarray(celebA_predictions_labels)

    print(celebA_predictions_numpy.shape)
    print(celebA_predictions_labels_numpy.shape)

    with open('CelebA_prediction_numpy', 'wb') as cApn:
        pkl.dump(celebA_predictions_numpy, cApn)

    with open('CelebA_predictions_numpy_labels', 'wb') as cApnL:
        pkl.dump(celebA_predictions_labels_numpy, cApnL)

finally:
    print('Done')
