from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import pandas as pd
import os
import pickle as pkl
from tqdm import tqdm

# Face Net github repository: https://github.com/timesler/facenet-pytorch
# To use this code you are required to run the following command in the terminal in your conda environment:
# pip install facenet-pytorch

# umap_learn project page: https://pypi.org/project/umap-learn/
# To use umap you must run the following command in the terminal in your conda environment:
# pip install umap-learn

# Initiate the deep learning models
# Resnet is the model actually used for getting new representations of the input images
resnet = InceptionResnetV1(pretrained='vggface2').eval()
resnet.classify = True

# Mtcnn is used for identifying where the face is in the input image and cropping the image
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)

# Uncropped root is the location of the unprocessed input images. The computer expects the images to be in directory
#      according to the identity which the belong to
input_image = "imageclassifier/image_data/Part_1/Brad_Pit/01.jpg"

image = Image.open(input_image)

image_cropped = mtcnn(image)

# Get the processed representation for the image
image_representation = resnet(image_cropped.unsqueeze(0)).detach().cpu()

# Flatten the representation and convert it to a numpy array for saving
image_representation_flat = image_representation.squeeze().numpy()
image_representation_flat = image_representation_flat.reshape(1,-1)

pickle_file = open("ClassiferPickel",'rb')
classifer_dict = pkl.load(pickle_file)
pickle_file.close()

print(input_image)
for name , each_classifier in classifer_dict.items():
    view_probability = pd.DataFrame(each_classifier.predict_proba(image_representation_flat), columns=each_classifier.classes_)
    print(name)
    print(view_probability.round(decimals=5))

