from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pandas as pd
import os
import pickle as pkl

# Face Net github repository: https://github.com/timesler/facenet-pytorch
# To use this code you are required to run the following command in the terminal in your conda environment:
# pip install facenet-pytorch

# Initiate the deep learning models
# Resnet is the model actually used for getting new representations of the input images
resnet = InceptionResnetV1(pretrained='vggface2').eval()
resnet.classify = True

# Mtcnn is used for identifying where the face is in the input image and cropping the image
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)

image_match_probability = []
input_image = "/home/teacheraccount/Downloads/Brad_Pit_Test_01.jpeg"

image = Image.open(input_image)

image_cropped = mtcnn(image)

# Get the processed representation for the image
image_representation = resnet(image_cropped.unsqueeze(0)).detach().cpu()

# Flatten the representation and convert it to a numpy array for saving
image_representation_flat = image_representation.squeeze().numpy()
image_representation_flat = image_representation_flat.reshape(1, -1)

pickle_file = open("ClassiferPickel", 'rb')
classifer_dict = pkl.load(pickle_file)
pickle_file.close()

print(input_image)
for name, each_classifier in classifer_dict.items():
    view_probability = pd.DataFrame(each_classifier.predict_proba(image_representation_flat),
                                    columns=each_classifier.classes_)
    probability = each_classifier.predict_proba(image_representation_flat)
    image_match_probability.append([name[0],probability[0][1]])
print(image_match_probability)
