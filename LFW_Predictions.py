import os
import pickle as pkl
import numpy as np
from tqdm import tqdm

# load classifiers from pickle saved from Doppelganger_Classifier
pickle_file = open("Saved_Classifiers.pkl", 'rb')
classifier_dict = pkl.load(pickle_file)
pickle_file.close()

sample_size = 10000
image_numpies_path = 'image_numpy_lfw'

# Empty list to collect all predictions for data set and labels for the predictions
image_predictions = []
image_predictions_labels = []



# to do smaller size set of data set sample_size above and uncomment the two line below
# dirs_list = sorted(os.listdir(image_numpies_path))
# folders = dirs_list[:sample_size]

# to do the whole data set uncomment the line below
folders = sorted(os.listdir(image_numpies_path))


# Navigate through each persons folder and run their image numpys through the doppelganger classifier
# and add to prediction list

for folder in tqdm(folders):
    image_match_probability = []
    image_file_path = os.path.join(image_numpies_path, folder)
    image_file_list = sorted(os.listdir(image_file_path))
    for file_name in image_file_list:
        image_data_path = os.path.join(image_file_path,file_name)
        image_data = np.load(image_data_path)
        image_data = image_data.reshape(1,-1)

        # this is the part that runs the numpy from CelebA through the doppelganger classifiers
        for name, each_classifier in classifier_dict.items():
            probability = each_classifier.predict_proba(image_data)
            image_match_probability.append(probability[0][1])
        image_predictions.append(image_match_probability)
        image_predictions_labels.append(folder)
# for each in image_predictions:
#     print(len(each))


try:
    predictions_numpy = np.asarray(image_predictions)
    predictions_labels_numpy = np.asarray(image_predictions_labels)

    print(predictions_numpy.shape)
    print(predictions_labels_numpy.shape)

    with open('LFW_prediction_numpy', 'wb') as data_file:
        pkl.dump(image_predictions, data_file)

    with open('LFW_predictions_numpy_labels', 'wb') as label_file:
        pkl.dump(image_predictions_labels, label_file)

finally:
    print('Done')
