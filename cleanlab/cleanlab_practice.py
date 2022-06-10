#!/usr/bin/env python

"""
This tutorial provides reproducible code to find the label errors for datasets:
MNIST, CIFAR-10, CIFAR-100, ImageNet, Caltech-256, Amazon Reviews, IMDB,
20News, and AudioSet. These datasets comprise 9 of the 10 datasets on
https://labelerrors.com .

Label errors are found using the pyx (predicted probs), pred (predicted labels),
and test label files, provided in this repo: (cleanlab/label-errors)

The QuickDraw dataset is excluded because the pyx file is 33GB and might
cause trouble on some machines. To find label errors in the QuickDraw dataset,
you can download the pyx file here:
https://github.com/cleanlab/label-errors/releases/tag/quickdraw-pyx-v1

This tutorial reproduces how we find the label errors on https://labelerrors.com
(prior to human validation on mTurk).

To more closely match the label errors on labelerrors.com and in the paper,
set reproduce_labelerrors_dot_com = True
"""

import cleanlab
import numpy as np
import json
import os
# from util import ALL_CLASSES
# To view the text data from labelerrors.com, we need:
from urllib.request import urlopen
# To view the image data from labelerrors.com, we need:
from skimage import io
from matplotlib import pyplot as plt

plt.interactive(False)

# Remove axes since we're plotting images, not graphs
rc = {"axes.spines.left" : False,
      "axes.spines.right" : False,
      "axes.spines.bottom" : False,
      "axes.spines.top" : False,
      "xtick.bottom" : False,
      "xtick.labelbottom" : False,
      "ytick.labelleft" : False,
      "ytick.left" : False}
plt.rcParams.update(rc)

# Find label errors in each dataset

# By default, the code below will use the most up-to-date theory and algorithms
# of confident learning, implemented in the cleanlab package.
# We recommend this for best results.
# However, if you need to more closely match the label errors 
# to match https://labelerrors.com, set `reproduce_labelerrors_dot_com = True`.
# There may be discrepancies in counts due to improvements to cleanlab
# since the work was published.

# Set to False for best/most-recent results (this approach also runs faster)
# Set to True to match the label errors on https://labelerrors.com
reproduce_labelerrors_website = False

# attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
#              'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
#               'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
#               'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
#               'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
#               'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

attributes = ['Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
              'Eyeglasses', 'Gray_Hair', 'Male', 'Mustache', 'No_Beard', 'Straight_Hair', 'Wavy_Hair',
              'Young']

for attribute in attributes:
    for cross_val_index in range(1, 4):
        title = f'Attribute: {attribute.capitalize()}, Index: {cross_val_index}'
        print('='*len(title), title, '='*len(title), sep='\n')

        # Get the cross-validated predicted probabilities on the test set.
        pyx = np.load(f"./pyx/{attribute}_{cross_val_index}_celeba_test_set_pyx.npy", allow_pickle=True)
        # Get the cross-validated predictions (argmax of pyx) on the test set.
        pred = np.load(f'./predicted_labels/{attribute}_{cross_val_index}_celeba_pyx_argmax_predicted_labels.npy', allow_pickle=True)
        # Get the test set labels
        labels = np.load(f'./original_label/{attribute}_{cross_val_index}_celeba_original_label.npy', allow_pickle=True)
        img_paths = np.load(f"./img_paths/{attribute}_{cross_val_index}_celeba_img_paths.npy", allow_pickle=True)

        # Find label error indices using cleanlab in one line of code.
        # This will use the most recent version of cleanlab with best results.
        # print('Finding label errors using cleanlab for {:,} '
        #       'examples and {} classes...'.format(*pyx.shape))
        label_error_indices = cleanlab.pruning.get_noise_indices(
            s=labels,
            psx=pyx,
            # Try prune_method='both' (C+NR in the confident learning paper)
            # 'both' finds fewer errors, but can sometimes increase precision
            prune_method='both',
            # num_to_remove_per_class=5,
            # multi_label=True,
            sorted_index_method='self_confidence',
        )
        num_errors = len(label_error_indices)
        print(f"Num Samples in {attribute}_{cross_val_index}: {img_paths.shape[0]}")
        print(f'Estimated number of errors in {attribute}_{cross_val_index}: {num_errors}')

        # Print an example
        # Grab the first label error found with cleanlab
        err_id = label_error_indices[0]

        # Custom code to visualize each label error from each dataset
        count = 0
        for index, id in enumerate(label_error_indices):
            given_label = labels[id]
            pred_label = pred[id]
            # if (labels[id] == 1):
            #     count += 1
            # if count==10:
            #     break
            url = img_paths[id]
            image = io.imread(url)  # read image data from a url
            plt.imshow(image, interpolation='nearest', aspect='auto', cmap='gray')
            plt.title(f"Attribute: {attribute}, Given: {given_label}, Pred: {pred_label}")
            plt.savefig(f"./label_error_figs/{attribute}_{cross_val_index}_givenLabel_{given_label}_{url[-10:-4]}.png")

            # print(f' * Label given in dataset: {given_label}')
            # print(f' * We Guess (argmax prediction) ((predicted label)): {pred_label}')
            # print(f' * Label Error Found: {url}\n')