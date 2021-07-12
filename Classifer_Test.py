import pickle
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

probability_dict = {}
pickle_file = open("ClassiferPickel",'rb')
classifer_dict = pickle.load(pickle_file)
pickle_file.close()

Celeb_Images = {'Pair_3':{'Beyonce':'imageclassifier/First5_numpy/Pair_3/Beyonce/04.jpg.npy',
                             'Solange':'imageclassifier/First5_numpy/Pair_3/Solange/00.jpg.npy'},
                   'Pair_4':{'Haley Duff':'imageclassifier/First5_numpy/Pair_4/Haley Duff/00.jpg.npy',
                             'Hilary':'imageclassifier/First5_numpy/Pair_4/Hilary/01.jpg.npy'},
                   'Pair_5':{'Ben Affleck':'imageclassifier/First5_numpy/Pair_5/Ben Affleck/00.jpg.npy',
                             'Casey Affleck':'imageclassifier/First5_numpy/Pair_5/Casey Affleck/00.jpg.npy'},
                   'Pair_6':{'Dave Franco':'imageclassifier/First5_numpy/Pair_6/Dave Franco/00.jpg.npy',
                             'James Franco':'imageclassifier/First5_numpy/Pair_6/James Franco/00.jpg.npy'},
                   'Pair_7':{'Brianna Cuoco':'imageclassifier/First5_numpy/Pair_7/Brianna Cuoco/01.jpg.npy',
                             'Kaley Cuoco':'imageclassifier/First5_numpy/Pair_7/Kaley Cuoco/00.jpg.npy'}
                   }
for each_pair in Celeb_Images.values():
    Celeb_Names = []
    Celeb_Image = []
    for name, test_image_path in each_pair.items():
        Celeb_Names.append(name)
        Celeb_Image.append(test_image_path)

    Celeb_one = Celeb_Names[0]
    image_one_path = np.load(Celeb_Image[0])
    image_one_path = image_one_path.reshape((1,-1))

    Celeb_two = Celeb_Names[1]
    image_two_path =np.load(Celeb_Image[1])
    image_two_path = image_two_path.reshape((1,-1))

    classifier_one = classifer_dict[Celeb_one]
    classifier_two = classifer_dict[Celeb_two]

    print("\n")
    print("________________________________________________________________________________________________")
    print("Target image: {} , Classifer for: {}".format(Celeb_one,Celeb_one))
    view_probability = pd.DataFrame(classifier_one.predict_proba(image_one_path), columns=classifier_one.classes_)
    print(view_probability.round(decimals=5))
    print("Target image: {} , Classifer for: {}".format(Celeb_two,Celeb_one))
    view_probability = pd.DataFrame(classifier_one.predict_proba(image_two_path), columns=classifier_one.classes_)
    print(view_probability.round(decimals=5))
    print("\n")
    print("________________________________________________________________________________________________")
    print("Target image: {} , Classifer for: {}".format(Celeb_one,Celeb_two))
    view_probability = pd.DataFrame(classifier_two.predict_proba(image_one_path), columns=classifier_two.classes_)
    print(view_probability.round(decimals=5))
    print("Target image: {} , Classifer for: {}".format(Celeb_two,Celeb_two))
    view_probability = pd.DataFrame(classifier_two.predict_proba(image_two_path), columns=classifier_two.classes_)
    print(view_probability.round(decimals=5))





