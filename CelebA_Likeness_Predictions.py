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

celebA_sample_size = 30000
celeba_numpies = os.listdir(celebA_numpies_path)[:celebA_sample_size]
celebA_predictions = []
celebA_predictions_labels = []

with open(celebA_labels_path, 'r') as labels_file:
    lines_read = labels_file.readlines()
    label_lines = lines_read[:celebA_sample_size]
    for line in tqdm(label_lines):
        celeba_line = line.split()
        file = celeba_line[0]
        label = celeba_line[1]
        numpy_file_name = file + ".npy"
        if numpy_file_name in celeba_numpies:
            # print(numpy_file_name)
            image_match_probability = []
            numpy_path = os.path.join(celebA_numpies_path, numpy_file_name)
            celebA_numpy = np.load(numpy_path)
            # print(celebA_numpy.shape)
            celebA_numpy = celebA_numpy.reshape(1, -1)
            # print(celebA_numpy.shape)

            for name, each_classifier in classifier_dict.items():
                probability = each_classifier.predict_proba(celebA_numpy)
                image_match_probability.append(probability[0][1])
                # print(name, len(image_match_probability))
                celebA_predictions.append(image_match_probability)
                celebA_predictions_labels.append(label)
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
