import os
import re

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.functional as TF

import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import torchmetrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb
import itertools
import shutil
import random
from tqdm import tqdm

# Helper function to show a batch
def show_landmarks_batch(batch):
    """Show image with landmarks for a batch of samples."""
    x, _ = batch
    batch_size = len(x)
    im_size = x.size(2)
    grid_border_size = 2

    grid = torchvision.utils.make_grid(x)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

def select_random_identity_pairs(num_pairs, identity_labels_path, identity_names_path, output_path):
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    identity_labels_df = pd.read_csv(identity_labels_path)
    identity_names_df = pd.read_csv(identity_names_path)
    unique_identities = identity_labels_df.identity_id.unique()

    for pair in range(num_pairs):
        random_pair = np.random.choice(unique_identities, size=2, replace=False).tolist()
        id_names = []
        for index, id in enumerate(random_pair):
            id_image_names = identity_labels_df[identity_labels_df.identity_id == id].image_name.values.tolist()
            id_names.append(identity_names_df[identity_names_df.image_id == id_image_names[0]].identity_name.values[0])

            if index == 0:
                output_data = list(zip(id_image_names, itertools.repeat(index, len(id_image_names))))
            else:
                output_data += zip(id_image_names, itertools.repeat(index, len(id_image_names)))
        output_df = pd.DataFrame(output_data, columns=["image_name", "label"])
        output_df.to_csv(output_path + f"{id_names[0]}_{id_names[1]}.csv", index=False)

def select_curated_identity_pairs(num_pairs, output_path, curated_pairs_path, identity_labels_path, identity_names_path):
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    identity_labels_df = pd.read_csv(identity_labels_path)
    curated_identities_df = pd.read_csv(curated_pairs_path)
    identity_names_df = pd.read_csv(identity_names_path)

    for category in curated_identities_df.columns:
        current_category_series = curated_identities_df[category].dropna()
        num_pairs = len(current_category_series.index)

        if category == "Mixed":
            num_to_select = 500
        else:
            num_to_select = 250

        for i in range(num_to_select):
            random_pair_index = random.randint(0, num_pairs)
            random_pair = current_category_series[random_pair_index]
            id1 = int(random_pair.split(",")[0])
            id2 = int(random_pair.split(",")[1])
            random_pair = [id1, id2]

            id_names = []
            for index, id in enumerate(random_pair):
                id_image_names = identity_labels_df[identity_labels_df.identity_id == id].image_name.values.tolist()
                id_names.append(identity_names_df[identity_names_df.image_id == id_image_names[0]].identity_name.values[0])

                if index == 0:
                    output_data = list(zip(id_image_names, itertools.repeat(index, len(id_image_names))))
                else:
                    output_data += zip(id_image_names, itertools.repeat(index, len(id_image_names)))
            output_df = pd.DataFrame(output_data, columns=["image_name", "label"])
            output_df.to_csv(output_path + f"{id_names[0]}_{id_names[1]}.csv", index=False)

class CelebA_Dataset_Pairs_Crossfold(Dataset):
    def __init__(self, path_to_current_pair_labels, split="training", random_seed=64, cross_val_index=1, transform=None):
        self.split=split

        self.path_to_images = "/home/nthom/Documents/datasets/CelebA/Img/img_align_celeba/"
        current_pair_df = pd.read_csv(path_to_current_pair_labels)
        current_pair_df = current_pair_df.sample(frac=1, random_state=random_seed)

        images_count = len(current_pair_df.index)
        all_image_names = current_pair_df.image_name.values.tolist()
        all_image_labels = current_pair_df.label.values.tolist()

        self.image_names = []
        self.image_labels = []

        # if self.split == "validation":
        #     split_coordinates = (int(images_count * 0.8), int(images_count * 0.9))
        if self.split == "test":
            if cross_val_index == 0:
                split_coordinates = (int(images_count * 0.75), images_count-1)
                for image in all_image_names[split_coordinates[0]:split_coordinates[1]]:
                    self.image_names.append(image)
                for label in all_image_labels[split_coordinates[0]:split_coordinates[1]]:
                    self.image_labels.append(label)
            elif cross_val_index == 1:
                split_coordinates = (0, int(images_count * 0.25))
                for image in all_image_names[split_coordinates[0]:split_coordinates[1]]:
                    self.image_names.append(image)
                for label in all_image_labels[split_coordinates[0]:split_coordinates[1]]:
                    self.image_labels.append(label)
            elif cross_val_index == 2:
                split_coordinates = (int(images_count * 0.25), int(images_count * 0.5))
                for image in all_image_names[split_coordinates[0]:split_coordinates[1]]:
                    self.image_names.append(image)
                for label in all_image_labels[split_coordinates[0]:split_coordinates[1]]:
                    self.image_labels.append(label)
            else:
                split_coordinates = (int(images_count * 0.5), int(images_count * 0.75))
                for image in all_image_names[split_coordinates[0]:split_coordinates[1]]:
                    self.image_names.append(image)
                for label in all_image_labels[split_coordinates[0]:split_coordinates[1]]:
                    self.image_labels.append(label)
        else:
            if cross_val_index == 0:
                split_coordinates = (0, int(images_count * 0.75))
                for image in all_image_names[split_coordinates[0]:split_coordinates[1]]:
                    self.image_names.append(image)
                for label in all_image_labels[split_coordinates[0]:split_coordinates[1]]:
                    self.image_labels.append(label)
            elif cross_val_index == 1:
                split_coordinates = (int(images_count * 0.25), images_count-1)
                for image_index, image in enumerate(all_image_names[split_coordinates[0]:split_coordinates[1]]):
                    self.image_names.append(image)
                for label in all_image_labels[split_coordinates[0]:split_coordinates[1]]:
                    self.image_labels.append(label)
            elif cross_val_index == 2:
                split_coordinates = (0, int(images_count * 0.25))
                for image in all_image_names[split_coordinates[0]:split_coordinates[1]]:
                    self.image_names.append(image)
                for label in all_image_labels[split_coordinates[0]:split_coordinates[1]]:
                    self.image_labels.append(label)

                split_coordinates = (int(images_count * 0.5), images_count-1)
                for image in all_image_names[split_coordinates[0]:split_coordinates[1]]:
                    self.image_names.append(image)
                for label in all_image_labels[split_coordinates[0]:split_coordinates[1]]:
                    self.image_labels.append(label)
            else:
                split_coordinates = (0, int(images_count * 0.5))
                for image in all_image_names[split_coordinates[0]:split_coordinates[1]]:
                    self.image_names.append(image)
                for label in all_image_labels[split_coordinates[0]:split_coordinates[1]]:
                    self.image_labels.append(label)

                split_coordinates = (int(images_count * 0.75), images_count - 1)
                for image in all_image_names[split_coordinates[0]:split_coordinates[1]]:
                    self.image_names.append(image)
                for label in all_image_labels[split_coordinates[0]:split_coordinates[1]]:
                    self.image_labels.append(label)
        # If there are any transform functions to be called, store them
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = self.path_to_images + self.image_names[idx]
        image = torchvision.io.read_image(img_path)
        image = TF.convert_image_dtype(image, torch.float)

        # Read in the attribute labels for the current input image
        label = self.image_labels[idx]

        # IF USING SOFTMAX
        # opposite_label = torch.ones(attributes.size()).to(torch.float32) - attributes
        # attributes = torch.tensor([attributes, opposite_label], dtype=torch.float32)

        # if self.split=="test":
        #     return image, label, img_path

        return image, label

class CelebA_Dataset_Pairs(Dataset):
    def __init__(self, path_to_current_pair_labels, split="training", random_seed=64, transform=None):
        self.split=split

        self.path_to_images = "/home/nthom/Documents/datasets/CelebA/Img/img_align_celeba/"
        current_pair_df = pd.read_csv(path_to_current_pair_labels)
        current_pair_df = current_pair_df.sample(frac=1, random_state=random_seed)

        images_count = len(current_pair_df.index)
        all_image_names = current_pair_df.image_name.values.tolist()
        all_image_labels = current_pair_df.label.values.tolist()

        self.image_names = []
        self.image_labels = []

        # if self.split == "validation":
        #     split_coordinates = (int(images_count * 0.8), int(images_count * 0.9))
        if self.split == "test":
            split_coordinates = (0, int(images_count * 0.25))
            for image in all_image_names[split_coordinates[0]:split_coordinates[1]]:
                self.image_names.append(image)
            for label in all_image_labels[split_coordinates[0]:split_coordinates[1]]:
                self.image_labels.append(label)
        else:
            split_coordinates = (int(images_count * 0.25), images_count-1)
            for image in all_image_names[split_coordinates[0]:split_coordinates[1]]:
                self.image_names.append(image)
            for label in all_image_labels[split_coordinates[0]:split_coordinates[1]]:
                self.image_labels.append(label)
        # If there are any transform functions to be called, store them
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = self.path_to_images + self.image_names[idx]
        image = torchvision.io.read_image(img_path)
        image = TF.convert_image_dtype(image, torch.float)

        # Read in the attribute labels for the current input image
        label = self.image_labels[idx]

        # IF USING SOFTMAX
        # opposite_label = torch.ones(attributes.size()).to(torch.float32) - attributes
        # attributes = torch.tensor([attributes, opposite_label], dtype=torch.float32)

        # if self.split=="test":
        #     return image, label, img_path

        return image, label

class Altered_MobileNet(LightningModule):
    def __init__(self, hyperparameters):
        super().__init__()
        # pretrained_mobilenetv3s = torchvision.models.mobilenet_v3_small(pretrained=True)
        # n_in_features = pretrained_mobilenetv3s.classifier[3].in_features
        # n_out_features = pretrained_mobilenetv3s.classifier[3].out_features  # get dimensions of fc l

        pretrained_mobilenetv3l = torchvision.models.mobilenet_v3_large(pretrained=True)
        n_in_features = pretrained_mobilenetv3l.classifier[3].in_features
        n_out_features = pretrained_mobilenetv3l.classifier[3].out_features  # get dimensions of fc l

        # pretrained_resnet18 = torchvision.models.resnet18(pretrained=True)
        # n_in_features = pretrained_resnet18.fc.in_features
        # n_out_features = pretrained_resnet18.fc.out_features

        # self.model = pretrained_mobilenetv3s
        # self.model.classifier[3] = nn.Linear(in_features=n_in_features, out_features=1, bias=True)
        self.model = pretrained_mobilenetv3l
        self.model.classifier[3] = nn.Linear(in_features=n_in_features, out_features=1, bias=True)
        # self.model = pretrained_resnet18
        # self.model.fc = nn.Linear(in_features=n_in_features, out_features=1, bias=True)

        self.sigmoid_layer = nn.Sigmoid()

        # self.pair = hyperparameters["pair"]
        # self.cross_val_index = hyperparamters["cross_val_index"]

        self.train_acc_metric = torchmetrics.Accuracy()
        self.val_acc_metric = torchmetrics.Accuracy()
        self.val_f1_metric = torchmetrics.F1Score()

        self.bce_criterion = torch.nn.BCELoss()

    def forward(self, x):
        h = self.model(x)
        h_sigmoid = self.sigmoid_layer(h)
        h_sigmoid = torch.squeeze(h_sigmoid)
        return h_sigmoid

    def training_step(self, train_batch, train_batch_idx):
        train_images, train_labels = train_batch

        h_sigmoid_train = self.forward(train_images)

        with torch.cuda.amp.autocast(enabled=False):
            train_loss = self.bce_criterion(h_sigmoid_train.to(torch.float32), train_labels.to(torch.float32))

        self.train_acc_metric(torch.gt(h_sigmoid_train, 0.49).to(torch.int), train_labels.to(torch.int))

        # self.log("Epoch", self.current_epoch, logger=True)
        # self.log("Batch", train_batch_idx, logger=True)
        self.log("Training Acc", self.train_acc_metric)
        self.log("Train Loss", train_loss)

        return train_loss

    def validation_step(self, val_batch, val_batch_idx):
        val_images, val_labels = val_batch

        h_sigmoid_val = self.forward(val_images)
        with torch.cuda.amp.autocast(enabled=False):
            val_loss = self.bce_criterion(h_sigmoid_val.to(torch.float32), val_labels.to(torch.float32))

        self.val_acc_metric(torch.gt(h_sigmoid_val, 0.49).to(torch.int), val_labels.to(torch.int))
        self.val_f1_metric(torch.gt(h_sigmoid_val, 0.49).to(torch.int), val_labels.to(torch.int))

        # self.log("Epoch", self.current_epoch, logger=True)
        # self.log("Batch", val_batch_idx, logger=True)
        self.log("Validation Acc", self.val_acc_metric)
        self.log("Validation F1", self.val_f1_metric)
        self.log("Validation Loss", val_loss)

        return val_loss

    def test_step(self, test_batch, test_batch_idx):
        test_images, test_labels, test_img_paths = test_batch

        # h_sigmoid_test = self.forward(test_images)
        # original = h_sigmoid_test.cpu()
        # opposite = torch.ones(h_sigmoid_test.cpu().size()).cpu() - original
        # original = h_softmax_test.cpu()
        # opposite = torch.ones(h_softmax_test.cpu().size()).cpu() - original
        # original = torch.unsqueeze(original, dim=1)
        # opposite = torch.unsqueeze(opposite, dim=1)
        # pyx = torch.cat([original, opposite], dim=-1)
        # labels = test_labels.to(torch.int).cpu()

        # IF USING SOFTMAX
        h_softmax_test = self.forward(test_images)
        pyx = h_softmax_test.cpu()
        labels = torch.argmax(test_labels.to(torch.int).cpu(), 1)

        preds = torch.gt(pyx, 0.49).to(torch.int)
        preds = torch.argmax(preds, 1)
        self.test_pyx_out.append(pyx)
        self.test_preds_out.append(preds)
        self.test_labels_out.append(labels)
        self.test_img_paths_out.append(test_img_paths)

    def test_epoch_end(self, outputs):
        test_pyx_out = torch.cat(self.test_pyx_out)
        test_preds_out = torch.cat(self.test_preds_out)
        test_labels_out = torch.cat(self.test_labels_out)
        test_img_paths_out = sum(self.test_img_paths_out, ())

        test_probs = np.save(f"./cleanlab/pyx/{self.attribute}_{self.cross_val_index}_celeba_test_set_pyx", test_pyx_out.numpy())
        test_preds = np.save(f"./cleanlab/predicted_labels/{self.attribute}_{self.cross_val_index}_celeba_pyx_argmax_predicted_labels", test_preds_out.numpy())
        test_labels = np.save(f"./cleanlab/original_label/{self.attribute}_{self.cross_val_index}_celeba_original_label", test_labels_out.numpy())
        test_img_paths = np.save(f"./cleanlab/img_paths/{self.attribute}_{self.cross_val_index}_celeba_img_paths", np.array(test_img_paths_out))

    def configure_optimizers(self):
        scheduler = None
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # "decay the learning rate with the cosine decay schedule without restarts"
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     patience=1,
        #     verbose=True
        # )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        if scheduler:
            # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "Validation Loss_epoch"}
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}

reload = False
path_to_saved_models = "./trained_models/"
saved_models = sorted(os.listdir(path_to_saved_models))

train = True
val_only = False
test = False

save = True

random_seed = 128
batch_size = 64
epochs = 4
gpus = -1
num_pairs = 1000
identity_labels_path="../preprocessed_data/pruned_by_num_samples/identity_CelebA_min-30.csv"
curated_identities_path="../preprocessed_data/sorted_dissimilar_identities/sorted_dissimilar_identities_samp_thresh-30_attribs-32_presence_20_comm_feat-85.csv"
identity_names_path="/home/nthom/Documents/datasets/CelebA/Anno/list_identity_celeba.csv"
output_path="../preprocessed_data/CelebA_pairs/"

hyperparamters = {}

pl.seed_everything(random_seed)
# select_random_identity_pairs(num_pairs, identity_labels_path, identity_names_path, output_path)
select_curated_identity_pairs(num_pairs, output_path, curated_identities_path, identity_labels_path, identity_names_path)

pairs = os.listdir(output_path)
for index, pair in enumerate(pairs):
    for cross_val_index in range(0, 4):
        random_seed = index * cross_val_index
        pl.seed_everything(random_seed)

        # saved_model_filename = f"{pair}_{cross_val_index}_mobilenetv3small_pretrained_CelebAAligned_noGender"
        # wandb_logger = WandbLogger(name=saved_model_filename, version=saved_model_filename, project="tom_pete_mobilenetv3small_dissimilar_noGender", entity='unr-mpl')
        saved_model_filename = f"{pair[:-4]}_{cross_val_index}_mobilenetv3large_pretrained_CelebAAligned" + "_0.001_StepLR_1"
        wandb_logger = WandbLogger(name=saved_model_filename, version=saved_model_filename, project="tom_pete", entity='unr-mpl')
        # saved_model_filename = f"{pair}_{cross_val_index}_resnet18_pretrained_CelebAAligned"
        # wandb_logger = WandbLogger(name=saved_model_filename, version=saved_model_filename, project="tom_pete_resnet18_dissimilar_noGender", entity='unr-mpl')

        model = Altered_MobileNet(hyperparamters)

        if train:
            train_dataset = CelebA_Dataset(
                path_to_current_pair_labels=output_path+pair,
                split="train",
                random_seed=random_seed,
                cross_val_index=cross_val_index,
                transform=None,
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)

            test_dataset = CelebA_Dataset(
                path_to_current_pair_labels=output_path + pair,
                split="test",
                random_seed=random_seed,
                cross_val_index=cross_val_index,
                transform=None,
            )
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)

            # for i_batch, sampled_batch in enumerate(test_loader):
            #     plt.figure()
            #     show_landmarks_batch(sampled_batch)
            #     plt.axis('off')
            #     plt.ioff()
            #     plt.show()

        else:
            test_dataset = CelebA_Dataset(
                path_to_current_pair_labels=output_path+pair,
                split="test",
                cross_val_index=cross_val_index,
                transform=None,
            )
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, drop_last=True, shuffle=False)


        checkpoint_callback = ModelCheckpoint(
            monitor='Validation Loss',
            dirpath=path_to_saved_models,
            filename='{epoch:02d}-{Validation Loss:.05f}-' + saved_model_filename,
            save_top_k=1,
            mode='min',
        )

        if train:
            trainer = pl.Trainer(
                logger=wandb_logger,
                log_every_n_steps=1,
                # precision=16,
                callbacks=[checkpoint_callback],
                strategy=DDPStrategy(find_unused_parameters=False),
                accelerator="auto",
                devices=-1,
                num_nodes=1,
                # limit_train_batches=0.05,
                # limit_val_batches=0.05,
                max_epochs=epochs
            )
        elif test:
            trainer = pl.Trainer(
                logger=wandb_logger,
                accelerator="auto",
                devices=1,
                num_nodes=1,
            )

        if save == False:
            trainer.enable_checkpointing = False

        if train == True:
            trainer.fit(model, train_loader, test_loader)
        else:
            trainer.test(model, test_loader)

        wandb.finish()
