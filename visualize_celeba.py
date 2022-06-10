import os

import torchvision, torch

import pandas as pd
import matplotlib.pyplot as plt

path_to_pairs = "./preprocessed_data/CelebA_pairs/"
path_to_output = "./preprocessed_data/CelebA_pairs_figs/"
path_to_images = "/home/nthom/Documents/datasets/CelebA/Img/img_align_celeba/"
pair_csvs = os.listdir(path_to_pairs)

for pair in pair_csvs:
    print(pair)
    current_pair_df = pd.read_csv(path_to_pairs + pair)
    image_names = current_pair_df.image_name.values.tolist()
    num_images = len(image_names)
    image_list = []
    for image in image_names:
        image_list.append(torch.unsqueeze(torchvision.io.read_image(path_to_images + image), dim=0))
    image_shape = image_list[0].shape
    image_tensor = torch.cat(image_list)
    grid = torchvision.utils.make_grid(image_tensor).numpy().transpose((1, 2, 0))
    plt.axis('off')
    plt.ioff()
    plt.imshow(grid)
    plt.savefig(path_to_output + pair[:-4] + ".png")