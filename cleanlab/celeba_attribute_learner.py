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

class CelebA_Dataset(Dataset):
    def __init__(self, split="training", attribute="Male", cross_val_index=1, transform=None):
        self.split=split

        path_to_image_labels = "/home/nthom/Documents/datasets/CelebA/Anno/list_attr_celeba.txt"
        self.path_to_images = "/home/nthom/Documents/datasets/CelebA/Img/img_align_celeba/"
        celeba_df = pd.read_csv(path_to_image_labels, sep=" ", skiprows=1)

        images_count = len(celeba_df.index)
        all_image_names = celeba_df.image_name.values.tolist()

        all_image_labels = celeba_df[attribute].values.tolist()

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
        # print(self.path_to_images + self.image_names[idx])
        img_path = self.path_to_images + self.image_names[idx]
        image = torchvision.io.read_image(img_path)
        image = TF.convert_image_dtype(image, torch.float)

        # Read in the attribute labels for the current input image
        # attributes = self.image_labels.iloc[idx,]
        attributes = self.image_labels[idx]
        attributes = torch.tensor(attributes)
        attributes = torch.gt(attributes, 0).to(torch.float32)
        # print(attributes)
        # exit(0)

        # IF USING SOFTMAX
        opposite_label = torch.ones(attributes.size()).to(torch.float32) - attributes
        attributes = torch.tensor([attributes, opposite_label], dtype=torch.float32)
        if self.split=="test":
            return image, attributes, img_path

        return image, attributes

class Altered_Resnet(LightningModule):
    def __init__(self, hyperparameters):
        super().__init__()
        # initialize ResNet
        pretrained_resnet = torchvision.models.resnet50(pretrained=True)
        self.n_features = pretrained_resnet.fc.out_features  # get dimensions of fc l

        # additional_layers = nn.Sequential(nn.Linear(
        #     self.n_features, self.n_features),
        #     nn.Linear(self.n_features, 1)
        # )

        # IF USING SOFTMAX
        additional_layers = nn.Sequential(
            nn.Linear(self.n_features, 2)
        )

        self.model = nn.Sequential(pretrained_resnet, additional_layers)

        self.sigmoid_layer = nn.Sigmoid()
        self.softmax_layer = nn.Softmax(dim=1)

        self.attribute = hyperparameters["attribute"]
        self.cross_val_index = hyperparamters["cross_val_index"]

        self.test_pyx_out = []
        self.test_preds_out = []
        self.test_labels_out = []
        self.test_img_paths_out = []

        self.train_acc_metric = torchmetrics.Accuracy()
        self.val_acc_metric = torchmetrics.Accuracy()

        self.bce_criterion = torch.nn.BCELoss()

    def forward(self, x):
        h = self.model(x)
        # h_sigmoid = self.sigmoid_layer(h)
        h_softmax = self.softmax_layer(h)
        # return h
        # return torch.squeeze(h_sigmoid)
        return h_softmax

    def training_step(self, train_batch, train_batch_idx):
        train_images, train_labels = train_batch

        # h_sigmoid_train = self.forward(train_images)
        h_softmax_train = self.forward(train_images)

        with torch.cuda.amp.autocast(enabled=False):
            # train_loss = self.bce_criterion(h_sigmoid_train.to(torch.float32), train_labels.to(torch.float32))
            train_loss = self.bce_criterion(h_softmax_train.to(torch.float32), train_labels.to(torch.float32))

        # self.train_acc_metric(torch.gt(h_sigmoid_train, 0.49).to(torch.int), train_labels.to(torch.int))
        self.train_acc_metric(torch.gt(h_softmax_train, 0.49).to(torch.int), train_labels.to(torch.int))

        # wandb.log({"Training Acc": self.train_acc_metric, "Train Loss": train_loss, "Epoch": self.current_epoch, "Batch": train_batch_idx})
        self.log("Epoch", self.current_epoch, logger=True)
        self.log("Batch", train_batch_idx, logger=True)
        self.log("Training Acc", self.train_acc_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Train Loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return train_loss

    def validation_step(self, val_batch, val_batch_idx):
        val_images, val_labels = val_batch

        # h_sigmoid_val = self.forward(val_images)
        h_softmax_val = self.forward(val_images)
        with torch.cuda.amp.autocast(enabled=False):
            # val_loss = self.bce_criterion(h_sigmoid_val.to(torch.float32), val_labels.to(torch.float32))
            val_loss = self.bce_criterion(h_softmax_val.to(torch.float32), val_labels.to(torch.float32))

        # self.val_acc_metric(torch.gt(h_sigmoid_val, 0.49).to(torch.int), val_labels.to(torch.int))
        self.val_acc_metric(torch.gt(h_softmax_val, 0.49).to(torch.int), val_labels.to(torch.int))

        # wandb.log({"Validation Acc": self.val_acc_metric, "Validation Loss": val_loss, "Epoch": self.current_epoch, "Batch": val_batch_idx})
        self.log("Epoch", self.current_epoch, logger=True)
        self.log("Batch", val_batch_idx, logger=True)
        self.log("Validation Acc", self.val_acc_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Validation Loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
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

# Configuration Variables
random_seed = 64
batch_size = 64
# attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
#              'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
#               'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
#               'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
#               'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
#               'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

attributes = ['Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
              'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
              'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
              'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

reload = True

path_to_saved_models = "/trained_models/"
saved_models = sorted(os.listdir(path_to_saved_models))

model_filename = f"epoch=00-Validation Loss=0.02927-No_Beard_resnet50_pretrained_softmax_CelebAAligned.ckpt"

hyperparamters = {}

train = False
val_only = False
test = True

save = False

epochs = 3
gpus = -1

pl.seed_everything(random_seed)

# logger_dict = {}
# for attribute in attributes:
#     for cross_val_index in range(1, 4):
#         saved_model_filename = f"{attribute}_{cross_val_index}_resnet50_pretrained_softmax_CelebAAligned"
#         logger_dict[f"{attribute}_{cross_val_index}"] = WandbLogger(name=saved_model_filename, project="cleanlab_celeba", entity='unr-mpl')

for attribute in attributes:
    for cross_val_index in range(1, 4):
        saved_model_filename = f"{attribute}_{cross_val_index}_resnet50_pretrained_softmax_CelebAAligned"
        # wandb.init(name=saved_model_filename, project="cleanlab_celeba", entity='unr-mpl')
        wandb_logger = WandbLogger(name=saved_model_filename, version=saved_model_filename, project="cleanlab_celeba", entity='unr-mpl')

        if reload:
            pattern = re.compile(f".{attribute}_{cross_val_index}.")
            for filename in saved_models:
                if pattern.search(filename):
                    print(f"Matched Filename: {filename}")
                    model_filename = filename
                    hyperparamters["attribute"] = attribute
                    hyperparamters["cross_val_index"] = cross_val_index
            model = Altered_Resnet.load_from_checkpoint(path_to_saved_models + model_filename, hyperparameters=hyperparamters)
            # model = nebullvm.optimize_torch_model(model, batch_size, input_sizes=[3, 218, 178])
        else:
            model = Altered_Resnet(hyperparamters)


        # print(model)
        # for i_batch, sampled_batch in enumerate(test_loader):
        #     plt.figure()
        #     show_landmarks_batch(sampled_batch)
        #     plt.axis('off')
        #     plt.ioff()
        #     plt.show()
        #     if i_batch == 10:
        #         print()

        if train:
            train_dataset = CelebA_Dataset(
                split="train",
                attribute=attribute,
                transform=None,
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)

            val_dataset = CelebA_Dataset(
                split="validation",
                attribute=attribute,
                cross_val_index=cross_val_index,
                transform=None,
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)
        elif val_only:
            val_dataset = CelebA_Dataset(
                split="validation",
                transform=None,
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)
        else:
            test_dataset = CelebA_Dataset(
                split="test",
                attribute=attribute,
                cross_val_index=cross_val_index,
                transform=None,
            )
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, drop_last=True, shuffle=False)


        checkpoint_callback = ModelCheckpoint(
            monitor='Validation Loss_epoch',
            dirpath=path_to_saved_models,
            filename='{epoch:02d}-{Validation Loss_epoch:.05f}-' + saved_model_filename,
            save_top_k=1,
            mode='min',
        )

        if train:
            trainer = pl.Trainer(
                logger=wandb_logger,
                # logger=logger_dict[f"{attribute}_{cross_val_index}"],
                # precision=16,
                callbacks=[checkpoint_callback],
                strategy=DDPStrategy(find_unused_parameters=False),
                accelerator="auto",
                devices=2,
                num_nodes=1,
                # limit_train_batches=0.05,
                # limit_val_batches=0.05,
                max_epochs=epochs
            )
        elif test:
            trainer = pl.Trainer(
                # logger=wandb_logger,
                accelerator="auto",
                devices=1,
                num_nodes=1,
            )

        if save == False:
            trainer.enable_checkpointing = False

        if train == True:
            trainer.fit(model, train_loader, val_loader)
            # trainer.fit(cl, train_loader)
        elif val_only == True:
            trainer.test(model, val_loader)
        else:
            trainer.test(model, test_loader)

        # logger_dict[f"{attribute}_{cross_val_index}"].finalize()
        # wandb_logger.finalize()
        wandb.finish()