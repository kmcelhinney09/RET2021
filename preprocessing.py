import os
import argparse
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg

class RET_Preprocessing():
    def __init__(self, args):
        self.args = args

        self.identity_labels_df = pd.read_csv(self.args.dataset_dir + self.args.identity_labels)
        self.attribute_labels_df = pd.read_csv(self.args.dataset_dir + self.args.attribute_labels, sep=" ", skiprows=1)

    def remove_df_rows(self, column_name, row_values, save_to_new=True, reload_df=True):
        temp_identity_labels_df = self.identity_labels_df.copy(deep=True)
        temp_attribute_labels_df = self.attribute_labels_df.copy(deep=True)

        for value in tqdm(row_values):
            image_names = temp_identity_labels_df[temp_identity_labels_df[column_name] == value]["image_name"].values.tolist()
            for image_name in image_names:
                temp_attribute_labels_df = temp_attribute_labels_df[temp_attribute_labels_df["image_name"] != image_name]

            temp_identity_labels_df = temp_identity_labels_df[temp_identity_labels_df[column_name] != value]

        if save_to_new:
            temp_attribute_labels_df.to_csv(self.args.preproc_save_path + f"list_attr_celeba_{self.args.under_rep_threshold}.txt", sep=" ")
            temp_identity_labels_df.to_csv(self.args.preproc_save_path + f"identity_CelebA_{self.args.under_rep_threshold}.csv")

        if reload_df:
            self.identity_labels_df = temp_identity_labels_df
            self.attribute_labels_df = temp_attribute_labels_df

    def compute_data_statistics(self):
        self.identity_counts = {}
        self.under_rep_identities = {}

        self.max = 0
        self.min = np.inf

        for identity in self.identity_labels_df.identity_id.unique():
            count = len(self.identity_labels_df[self.identity_labels_df["identity_id"] == identity].index)

            if count > self.max:
                self.max = count
            if count < self.min:
                self.min = count

            if count <= self.args.under_rep_threshold:
                self.under_rep_identities[identity] = count
            self.identity_counts[identity] = count

        self.remove_df_rows("identity_id", list(self.under_rep_identities.keys()))
        print()

    # def find_agreeable_attributes(self):

if __name__ == "__main__":
    config_path = "./config.yaml"

    parser = argparse.ArgumentParser(description="smart_attacker_unsupervised")
    yaml_config = yaml_config_hook(config_path)

    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    preprocessing_object = RET_Preprocessing(args)

    preprocessing_object.compute_data_statistics()
    preprocessing_object.compute_data_statistics()
    print()
