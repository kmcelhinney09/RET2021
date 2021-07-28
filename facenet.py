from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import pickle as pkl
import umap
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt

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
uncropped_root = "./lfw/"

# Cropped root is the location that you wish to save the cropped images of each identity
cropped_root = "./cropped_images_lfw/"

# Pickled root is the location that you wish to save the processed representations of the input images
numpy_root = "./image_numpy_lfw/"

# Create directories for output directories if they do not exist
if not os.path.exists(cropped_root):
    os.makedirs(cropped_root)
if not os.path.exists(numpy_root):
    os.makedirs(numpy_root)

# Identities list is a list of all the identity names found in the uncropped root folder
identities_list = sorted(os.listdir(uncropped_root))

# Identity dict is where all of the output file paths are saved
identity_dict = {}

# Each of these lists is used to save information which is used for plotting the UMAP/TSNE representations of raw images
#      vs the processed representations
representations = []
images = []
classes = []

# Iterate through the identities
for index, identity in enumerate(identities_list):
    # Create output directories for each identity if they do not exist
    if not os.path.exists(cropped_root + identity):
        os.makedirs(cropped_root + identity)
    if not os.path.exists(numpy_root + identity):
        os.makedirs(numpy_root + identity)

    # Create lists in the identity dictionary for each of the important file paths
    identity_dict[identity] = {"uncropped_path": [], "cropped_path": [], "pickle_path": []}
    print(identity)
    # Iterate through each image for each identity
    for file in tqdm(os.listdir(uncropped_root+identity)):

        # Append the pertinent file paths to each list 
        identity_dict[identity]["uncropped_path"].append(uncropped_root+identity+'/'+file)
        identity_dict[identity]["cropped_path"].append(cropped_root+identity+'/'+file)
        identity_dict[identity]["pickle_path"].append(numpy_root+identity+'/'+file)

        # Open the current uncropped image
        image = Image.open(uncropped_root+identity+'/'+file)

        # Crop the uncropped image and save the file
        image_cropped = mtcnn(image, save_path=cropped_root + identity + '/' + file)

        # Alternatively crop the image without saving the file        
        # image_cropped = mtcnn(image)
        if image_cropped == None:
            continue

        # Get the processed representation for the image
        image_representation = resnet(image_cropped.unsqueeze(0)).detach().cpu()
        
        # Flatten the representation and convert it to a numpy array for saving
        image_representation_flat = image_representation.squeeze().numpy()

        # Save the numpy array of the learned representation
        np.save(numpy_root + identity + '/' + file, image_representation_flat, allow_pickle=True)

        # # Append the representation for plotting
        # representations.append(image_representation_flat.tolist())
        #
        # # Append the "class" or identity label for plotting
        # classes.append(index)
        #
        # # Append a flattened version of the cropped input image for plotting
        # images.append(np.array(image_cropped).flatten())

    identity_df = pd.DataFrame(identity_dict[identity])

#
# representations = np.array(representations)
# classes = np.array(classes)
#
# # Initialize umap and tsne
# umap_reducer = umap.UMAP()
# tsne_reducer = TSNE()
#
# umap_embedding = umap_reducer.fit_transform(representations)
#
# plt.scatter(
#     umap_embedding[:, 0],
#     umap_embedding[:, 1],
#     c=classes)
#
# plt.show()
#
# tsne_embedding = tsne_reducer.fit_transform(representations)
#
# plt.scatter(
#     tsne_embedding[:, 0],
#     tsne_embedding[:, 1],
#     c=classes)
#
# plt.show()
#
#
# #####
#
# umap_embedding = umap_reducer.fit_transform(images)
#
# plt.scatter(
#     umap_embedding[:, 0],
#     umap_embedding[:, 1],
#     c=classes)
#
# plt.show()
#
# tsne_embedding = tsne_reducer.fit_transform(images)
#
# plt.scatter(
#     tsne_embedding[:, 0],
#     tsne_embedding[:, 1],
#     c=classes)
#
# plt.show()
#
