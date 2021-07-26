from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pandas as pd
import os
import pickle as pkl

# load classifiers from pickle saved from Doppelganger_Classifier
pickle_file = open("Saved_Classifiers.pkl", 'rb')
classifer_dict = pkl.load(pickle_file)
pickle_file.close()

#file path to celebA numpys created using facenet and their associated labels document
celebA_labels_path = 'identity_CelebA.txt'
celebA_numpies_path = 'img_align_celeba_numpy'


celeba_numpies = os.listdir(celebA_numpies_path)
print(celeba_numpies)
celebA_labeles_dict = {}
image_match_probability = []


with open(celebA_labels_path, 'r') as labels_file:
    label_lines = labels_file.readlines()
    for line in label_lines:
        celeba_line = line.split()
        file = celeba_line[0]
        label = celeba_line[1]
        celebA_labeles_dict[file] = label
        numpy_file_name = file + ".npy"
        if numpy_file_name in celeba_numpies:
            numpy_path = os.path.join(celebA_numpies_path,numpy_file_name)
            for name, each_classifier in classifer_dict.items():
                probability = each_classifier.predict_proba(numpy_path)
                image_match_probability.append([name[0],probability[0][1]])

