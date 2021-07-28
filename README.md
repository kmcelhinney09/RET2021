# RET2021
This is work done by the Biometrics group for the  Research Experience For Teachers at UNR in the Summer of 2021 in the 
University of Nevada Reno Computer Perceptions Lab. 

## Description

	The series of scripts are used to take in a set of celeb image data, crop out their face using MTCNN, and extract facial features utilizing InceptionV1 both from  facenet_pytorch. The resulting numpys are then used to train an MLPClassifier from sklearn on telling the difference between one celebrity and their celebrity doppelganger, for example, Chris O'Donnell versus Stephen Amell.  With the current image set in the downloads folder, a total of 149 doppelganger classifiers can be trained. 	
	These doppelganger classifiers are then used to generate a probability that an input image, that has been processed through MTCNN and InceptionV1, looks like the target celeb of the classifier. So from the example above the Chris/Stephen classifier is used to output the probability that the input image is Chris O'Donnell. The 149 probabilities are then put into a python list to create a likeness feature for the input image.
	The likeness features are then run through a verification script using CelebA to see if we can identify the image based on the 149 likeness features.


## Getting Started
To get started all you will need to do is clone this repository. All scripts and file navigation is based on the way the file structure is in GitHub. 

1)  You first need to run facenet.py on the download folder, placing the path for the download folder next to the uncropped_root. This will create the NumPy arrays of the facial features needed to move on.
2) Run the Doppelganger_classifier.py. This is designed to go through the NumPy arrays created above and train the MLP classifiers for the doppelganger pairs. This will create a pickle of a python dictionary that has the two celebrity names from the pair as the key and the MLP model as the value. The GUI will use this to classify an input image taken from the webcam

From here you can use the other scripts to do other classifications.


### Dependencies
This was built using pycharm  running Python 3.9 and the following modules need to be downloaded:
1) Numpy
2) tqdm
3) pip install facenet-pytorch
4) Sklearn
5) umap
6)OpenCv
7) Pillow
8)PySimpleGui

### Executing program
Each script is executed from the  IDE

## Authors

Kevin McElhinney,
Michael Dambach,
Amy White,
Nate Thom

## Version History
