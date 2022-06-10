import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import functools
from multiprocessing import Pool
import time
from PIL import Image
import psutil
import math
import matplotlib.pyplot as plt

def determine_presence_of_attribute(identity, num_pos_attr_per_identity, positive_attr_count_dict):
    # Create list that will ultimately be output
    current_identity_list = [identity]
    # Get all rows in identity/image name dataframe for which the "identity_id" column is equal to the input identity parameter.
    # The convert all of these values into a list
    current_identity_image_names = identity_labels_df[identity_labels_df['identity_id'] == identity].image_name.values.tolist()
    # Get all rows in the image name/attribute label dataframe for which some value in the current_identity_image_names is equal to a value in the "image_name" column
    current_identity_attributes_df = attribute_labels_df[attribute_labels_df.image_name.isin(current_identity_image_names)]
    current_identity_attributes_df_length = len(current_identity_attributes_df.index)
    # Iterate over each attribute in the current_identity_attributes_df
    for column in current_identity_attributes_df.drop("image_name", axis=1):
        # Create a list that represents the current column
        current_column_df = current_identity_attributes_df[column]
        # Get the total number of samples in the column (this could be moved out of the for loop)
        # Get all of the positive values
        current_column_df = current_column_df[current_column_df.isin([1])]

        # Compute a ratio of positive to negative values and mark the attribute as existing in the identity if the ratio is over some threshold value
        ratio = (len(current_column_df.index)/ current_identity_attributes_df_length)
        if ratio >= 0.1:
            current_identity_list.append(1)
            num_pos_attr_per_identity += 1
            positive_attr_count_dict[column] += 1
        else:
            current_identity_list.append(0)
    return current_identity_list, num_pos_attr_per_identity

# def determine_similar_identities(attribute_count_dict: 0 , identity):
def determine_similar_identities(identity):
    current_identity_df = identity_attributes_df[identity_attributes_df.identity_id == identity]
    current_identity_column_values = current_identity_df.drop("identity_id", axis=1).values.tolist()

    current_identity_similar_identities_list = [identity]

    identities_df = identity_attributes_df[identity_attributes_df['identity_id'] != identity]

    # num_attributes = len(attributes_to_keep)
    num_positive_attributes = current_identity_df.drop("identity_id", axis=1).sum().sum()
    for other_identity in identities_df.identity_id.unique():
        num_agreeing_attributes = 0
        other_identity_df = identities_df[identities_df.identity_id == other_identity]
        other_identity_column_values = other_identity_df.drop("identity_id", axis=1).values.tolist()
        for index, column in enumerate(attributes_to_keep):
            if (current_identity_column_values[0][index] == 1 and other_identity_column_values[0][index] == 1):
                # attribute_count_dict[column] += 1
                num_agreeing_attributes += 1
        ratio = num_agreeing_attributes / num_positive_attributes
        if ratio >= .10:
            current_identity_similar_identities_list.append(other_identity_df.identity_id.values.tolist()[0])
    return current_identity_similar_identities_list
    # return attribute_count_dict

def determine_dissimilar_identities(identity):
    current_identity_df = identity_attributes_df[identity_attributes_df.identity_id == identity]
    current_identity_column_values = current_identity_df.drop(labels="identity_id", axis=1).values.tolist()

    current_identity_similar_identities_list = [identity]

    identities_df = identity_attributes_df[identity_attributes_df['identity_id'] != identity]

    # num_attributes = len(attributes_to_keep)
    num_positive_attributes = current_identity_df.drop(labels="identity_id", axis=1).sum().sum()
    for other_identity in identities_df.identity_id.unique():
        num_disagreeing_attributes = 0
        other_identity_df = identities_df[identities_df.identity_id == other_identity]
        other_identity_column_values = other_identity_df.drop("identity_id", axis=1).values.tolist()
        for index, column in enumerate(attributes_to_keep):
            if ((current_identity_column_values[0][index] - other_identity_column_values[0][index]) != 0):
                # attribute_count_dict[column] += 1
                num_disagreeing_attributes += 1
        ratio = num_disagreeing_attributes / num_positive_attributes
        if ratio >= .90:
            current_identity_similar_identities_list.append(other_identity_df.identity_id.values.tolist()[0])
    return current_identity_similar_identities_list
    # return attribute_count_dict

if __name__=="__main__":
    # attributes_to_keep = ['Arched_Eyebrows', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Gray_Hair', 'Male', 'Mustache', 'Narrow_Eyes', 'Receding_Hairline', 'Straight_Hair', 'Wavy_Hair']
    # attributes_to_keep = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    attributes_to_keep = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
                          'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                          'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
                          'High_Cheekbones', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
                          'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
                          'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
                          'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    # Data Options
    path_to_images = "/home/nthom/Documents/datasets/CelebA/Img/img_align_celeba/"
    # identity_labels_path = "/home/nthom/Documents/datasets/CelebA/Anno/identity_CelebA.csv"
    # attribute_labels_path = "/home/nthom/Documents/datasets/CelebA/Anno/list_attr_celeba.txt"
    identity_labels_path = "./preprocessed_data/pruned_by_num_samples/identity_CelebA_min-30.csv"
    attribute_labels_path = "./preprocessed_data/pruned_by_num_samples/list_attr_celeba_min-30.csv"

    # Preprocessing Options
    preproc_save_path = "./preprocessed_data/"

    # Read in data frames
    identity_labels_df = pd.read_csv(identity_labels_path)
    # attribute_labels_df = pd.read_csv(attribute_labels_path, sep=" ", skiprows=1)
    attribute_labels_df = pd.read_csv(attribute_labels_path)

    # Determine presence of attribute code block
    #**********#
    print("Determining presence of attributes...")

    # with Pool(16) as p:
    #     identity_attributes_list = p.map(determine_presence_of_attribute, identity_labels_df.identity_id.unique())

    positive_attr_count_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 0, 'Attractive': 0, 'Bags_Under_Eyes': 0, 'Bald': 0, 'Bangs': 0, 'Big_Lips': 0, 'Big_Nose': 0, 'Black_Hair': 0, 'Blond_Hair': 0, 'Blurry': 0, 'Brown_Hair': 0, 'Bushy_Eyebrows': 0, 'Chubby': 0, 'Double_Chin': 0, 'Eyeglasses': 0, 'Goatee': 0, 'Gray_Hair': 0, 'Heavy_Makeup': 0, 'High_Cheekbones': 0, 'Male': 0, 'Mouth_Slightly_Open': 0, 'Mustache': 0 , 'Narrow_Eyes': 0 , 'No_Beard': 0 , 'Oval_Face': 0 , 'Pale_Skin': 0 , 'Pointy_Nose': 0 , 'Receding_Hairline': 0 , 'Rosy_Cheeks': 0 , 'Sideburns': 0 , 'Smiling': 0 , 'Straight_Hair': 0 , 'Wavy_Hair': 0 , 'Wearing_Earrings': 0 , 'Wearing_Hat': 0 , 'Wearing_Lipstick': 0 , 'Wearing_Necklace': 0 , 'Wearing_Necktie': 0 , 'Young': 0}

    num_pos_attr_per_identity = 0
    identity_attribute_list = []
    for id in tqdm(identity_labels_df.identity_id.unique()):
        output_list, num_pos_attr_per_identity = determine_presence_of_attribute(id, num_pos_attr_per_identity, positive_attr_count_dict)
        identity_attribute_list.append(output_list)
    num_pos_attr_per_identity /= len(identity_labels_df.identity_id.unique())

    print("Done.")
    #**********#

    # # Determine similar identities code block
    # #**********#
    #
    # identity_attributes_df = pd.DataFrame(identity_attribute_list, columns=["identity_id", '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'])
    #
    # for column in identity_attributes_df.drop(labels="identity_id", axis=1):
    #     if column not in attributes_to_keep:
    #         identity_attributes_df = identity_attributes_df.drop(labels=column, axis=1)
    #
    # # attribute_count_dict = {'Arched_Eyebrows': 0, 'Bald': 0, 'Bangs': 0, 'Black_Hair': 0, 'Blond_Hair': 0, 'Brown_Hair': 0, 'Chubby': 0, 'Double_Chin': 0, 'Eyeglasses': 0, 'Gray_Hair': 0, 'Male': 0, 'Mustache': 0, 'Narrow_Eyes': 0, 'Receding_Hairline': 0, 'Straight_Hair': 0, 'Wavy_Hair': 0}
    #
    # # func = functools.partial(determine_similar_identities, attribute_count_dict)
    #
    # # start_time = time.time()
    # # for id in tqdm(identity_attributes_df.identity_id.unique()[:16]):
    # #     func(id)
    # # print(f"Run Time: {time.time() - start_time}")
    #
    # start_time = time.time()
    # with Pool(16) as p:
    #     # attribute_count_list = p.map(func, identity_attributes_df.identity_id.unique())
    #     similar_identities_list = p.map(determine_similar_identities, identity_attributes_df.identity_id.unique())
    # print(f"Run Time: {time.time() - start_time}")
    #
    # similar_identities_df = pd.DataFrame(similar_identities_list)
    # similar_identities_df = similar_identities_df.T
    # similar_identities_df_header = similar_identities_df.iloc[0]
    # similar_identities_df = similar_identities_df[1:]
    # similar_identities_df.columns = similar_identities_df_header
    # similar_identities_df_rows = similar_identities_df.shape[0]
    # similar_identities_df_columns = similar_identities_df.shape[1]
    #
    # # Print similar identity statistics
    # print(f"Num Pairs: {similar_identities_df.count().sum()}")
    #
    # num_id_w_zero = 0
    # for column in similar_identities_df:
    #     if (len(similar_identities_df.index) - similar_identities_df[column].isnull().sum()) == 0:
    #         num_id_w_zero += 1
    # print(f'Number of IDs: {similar_identities_df.shape[1]}')
    # print(f"Num IDs w/ No Pairs: {num_id_w_zero}")
    # print(f"Ratio: {num_id_w_zero / similar_identities_df.shape[1]}")
    #
    # # Save similar identities dataframe as csv
    # similar_identities_df.to_csv(
    #     "./preprocessed_data/similar_identities/similar_identities_samp_thresh-30_attribs-40_intersim-25_comm_feat-100.csv",
    #     index=False)
    #
    # # Read similar identities csv into dataframe
    # similar_identities_df = pd.read_csv("./preprocessed_data/similar_identities/similar_identities_samp_thresh-30_attribs-40_intersim-25_comm_feat-100.csv")
    #
    # # Create and plot similar identity grids
    # column_identity_images_list = []
    # value_identity_images_list = []
    # for column in similar_identities_df:
    #     current_column_df = similar_identities_df[column]
    #     if (current_column_df.isnull().sum()) == 821:
    #         continue
    #     column_identity_images_list = identity_labels_df[identity_labels_df['identity_id'] == int(float(column))].image_name.values.tolist()
    #     current_column_im = Image.open(path_to_images + column_identity_images_list[0])
    #     for value in current_column_df:
    #         value_identity_images_list = identity_labels_df[identity_labels_df['identity_id'] == int(float(value))].image_name.values.tolist()
    #
    #         images_list = []
    #
    #         for image in column_identity_images_list:
    #             images_list.append(image)
    #         for image in value_identity_images_list:
    #             images_list.append(image)
    #         images_count = len(images_list)
    #         grid_size = math.ceil(math.sqrt(images_count))
    #         fig, axes = plt.subplots(grid_size, grid_size, figsize=(40, 40))
    #         current_file_number=0
    #         for image_filename in images_list:
    #             x_position = current_file_number % grid_size
    #             y_position = current_file_number // grid_size
    #             plt_image = plt.imread(path_to_images + images_list[current_file_number])
    #             axes[x_position, y_position].imshow(plt_image)
    #             current_file_number += 1
    #         plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    #         plt.savefig(f"./preprocessed_data/similar_identity_grids/{column}-{value}.png")
    #         break
    #
    #         # value_column_im = Image.open(path_to_images + value_identity_images_list[0])
    #         # current_column_im.show()
    #         # value_column_im.show()
    #         # x=1
    #         # for proc in psutil.process_iter():
    #         #     if proc.name() == "display":
    #         #         proc.kill()
    #
    # #**********#

    # #**********#
    # print("Finding dissimilar identities...")
    #
    identity_attributes_df = pd.DataFrame(identity_attribute_list, columns=["identity_id", '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'])
    #
    # for column in identity_attributes_df.drop("identity_id", axis=1):
    #     if column not in attributes_to_keep:
    #         identity_attributes_df = identity_attributes_df.drop(column, axis=1)
    #
    # # attribute_count_dict = {'Arched_Eyebrows': 0, 'Bald': 0, 'Bangs': 0, 'Black_Hair': 0, 'Blond_Hair': 0, 'Brown_Hair': 0, 'Chubby': 0, 'Double_Chin': 0, 'Eyeglasses': 0, 'Gray_Hair': 0, 'Male': 0, 'Mustache': 0, 'Narrow_Eyes': 0, 'Receding_Hairline': 0, 'Straight_Hair': 0, 'Wavy_Hair': 0}
    #
    # # func = functools.partial(determine_similar_identities, attribute_count_dict)
    #
    # # start_time = time.time()
    # # for id in tqdm(identity_attributes_df.identity_id.unique()[:16]):
    # #     func(id)
    # # print(f"Run Time: {time.time() - start_time}")
    #
    # start_time = time.time()
    # with Pool(16) as p:
    #     # attribute_count_list = p.map(func, identity_attributes_df.identity_id.unique())
    #     dissimilar_identities_list = p.map(determine_dissimilar_identities, identity_attributes_df.identity_id.unique())
    # print(f"Total Run Time: {time.time() - start_time}")
    #
    # dissimilar_identities_df = pd.DataFrame(dissimilar_identities_list)
    # dissimilar_identities_df = dissimilar_identities_df.T
    # dissimilar_identities_df_header = dissimilar_identities_df.iloc[0]
    # dissimilar_identities_df = dissimilar_identities_df[1:]
    # dissimilar_identities_df.columns = dissimilar_identities_df_header
    # dissimilar_identities_df_rows = dissimilar_identities_df.shape[0]
    # dissimilar_identities_df_columns = dissimilar_identities_df.shape[1]
    #
    # print("Done.")
    #
    # # Compute dissimilar pair statistics
    # print(f"Num Pairs: {dissimilar_identities_df.count().sum()}")
    #
    # num_id_w_zero = 0
    # num_id_w_lte2 = 0
    # for column in dissimilar_identities_df:
    #     if (len(dissimilar_identities_df.index) - dissimilar_identities_df[column].isnull().sum()) == 0:
    #         num_id_w_zero += 1
    #     if (len(dissimilar_identities_df.index) - dissimilar_identities_df[column].isnull().sum()) <= 2:
    #         num_id_w_lte2 += 1
    # print(f'Number of IDs: {dissimilar_identities_df.shape[1]}')
    # print(f"Num IDs w/ No Pairs: {num_id_w_zero}")
    # print(f"Num IDs w/ Less Than 2 Pairs: {num_id_w_lte2}")
    #
    #
    # # Save dissimilar identities dataframe to csv
    # dissimilar_identities_df.to_csv("./preprocessed_data/dissimilar_identities/dissimilar_identities_samp_thresh-30_attribs-39_intersim-10_comm_feat-90.csv", index=False)

    dissimilar_identities_df = pd.read_csv("./preprocessed_data/dissimilar_identities/dissimilar_identities_samp_thresh-30_attribs-39_intersim-10_comm_feat-90.csv")

    for column in dissimilar_identities_df:
        print(f"Column: {column}")

        # for value in dissimilar_identities_df:
        #     print(f"Value: {value}")
            # print(identity_attributes_df[identity_attributes_df.identity_id==column])

    # Compute dissimilar pair statistics
    print(f"Num Pairs: {dissimilar_identities_df.count().sum()}")

    num_id_w_zero = 0
    num_id_w_lte2 = 0
    for column in dissimilar_identities_df:
        if (len(dissimilar_identities_df.index) - dissimilar_identities_df[column].isnull().sum()) == 0:
            num_id_w_zero += 1
        if (len(dissimilar_identities_df.index) - dissimilar_identities_df[column].isnull().sum()) <= 2:
            num_id_w_lte2 += 1

    print(f'Number of IDs: {dissimilar_identities_df.shape[1]}')
    print(f"Num IDs w/ No Pairs: {num_id_w_zero}")
    print(f"Num IDs w/ Less Than 2 Pairs: {num_id_w_lte2}")