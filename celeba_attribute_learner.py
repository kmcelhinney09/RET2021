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
    def __init__(self, split="training", attribute="Male", transform=None):
        self.split=split

        path_to_image_labels = "/home/nthom/Documents/datasets/CelebA/Anno/list_attr_celeba.txt"
        self.path_to_images = "/home/nthom/Documents/datasets/CelebA/Img/img_align_celeba/"
        celeba_df = pd.read_csv(path_to_image_labels, sep=" ", skiprows=1)
        images_count = len(celeba_df.index)
        all_image_names = celeba_df.image_name.values.tolist()

        all_image_labels = celeba_df[attribute]

        # train_split = (celeba_images_df.image_name[0], celeba_images_df.image_name[int(images_count * 0.8)])
        # val_split = (train_split[1], celeba_images_df.image_name[int(images_count * 0.9)])
        # test_split = (val_split[1], celeba_images_df.image_name[images_count-1])

        print(f"Images Count: {images_count}")

        if self.split == "validation":
            split_coordinates = (int(images_count * 0.8), int(images_count * 0.9))
        elif self.split == "test":
            split_coordinates = (int(images_count * 0.9), images_count-1)
        else:
            split_coordinates = (0, int(images_count * 0.8))

        self.image_names = all_image_names[split_coordinates[0]:split_coordinates[1]]
        self.image_labels = all_image_labels[split_coordinates[0]:split_coordinates[1]]

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
        attributes = self.image_labels.iloc[idx,]
        attributes = torch.tensor(attributes)
        attributes = torch.gt(attributes, 0).to(torch.float32)

        if self.split=="test":
            return image, attributes, img_path

        return image, attributes

class Altered_Resnet(LightningModule):
    def __init__(self, hyperparameters):
        super().__init__()
        # initialize ResNet
        self.model = torchvision.models.resnet50(pretrained=False)
        self.n_features = self.model.fc.in_features  # get dimensions of fc layer
        self.model.fc = nn.Linear(self.n_features, 1)
        self.sigmoid_layer = nn.Sigmoid()

        self.test_pyx_out = []
        self.test_preds_out = []
        self.test_labels_out = []
        self.test_img_paths_out = []

        self.train_acc_metric = torchmetrics.Accuracy()
        self.val_acc_metric = torchmetrics.Accuracy()

        self.bce_criterion = torch.nn.BCELoss()

    def forward(self, x):
        h = self.model(x)
        h_sigmoid = self.sigmoid_layer(h)
        # return h
        return torch.squeeze(h_sigmoid)

    def training_step(self, train_batch, train_batch_idx):
        train_images, train_labels = train_batch

        h_sigmoid_train = self.forward(train_images)

        with torch.cuda.amp.autocast(enabled=False):
         train_loss = self.bce_criterion(h_sigmoid_train.to(torch.float32), train_labels.to(torch.float32))

        self.train_acc_metric(torch.gt(h_sigmoid_train, 0.49).to(torch.int), train_labels.to(torch.int))
        self.log("Training Acc", self.train_acc_metric, on_step=True, on_epoch=True, prog_bar=True)
        self.log("Train Loss", train_loss, on_step=True, on_epoch=True, prog_bar=True)

        return train_loss

    def validation_step(self, val_batch, val_batch_idx):
        val_images, val_labels = val_batch

        h_sigmoid_val = self.forward(val_images)
        with torch.cuda.amp.autocast(enabled=False):
            val_loss = self.bce_criterion(h_sigmoid_val.to(torch.float32), val_labels.to(torch.float32))

        self.val_acc_metric(torch.gt(h_sigmoid_val, 0.49).to(torch.int), val_labels.to(torch.int))
        self.log("Validation Acc", self.val_acc_metric, on_step=True, on_epoch=True, prog_bar=True)
        self.log("Validation Loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)

        return val_loss

    def test_step(self, test_batch, test_batch_idx):
        test_images, test_labels, test_img_paths = test_batch

        h_sigmoid_test = self.forward(test_images)

        original = h_sigmoid_test.cpu()
        opposite = torch.ones(h_sigmoid_test.cpu().size()).cpu() - original
        original = torch.unsqueeze(original, dim=1)
        opposite = torch.unsqueeze(opposite, dim=1)
        pyx = torch.cat([original, opposite], dim=-1)
        preds = torch.gt(pyx, 0.49).to(torch.int)
        preds = torch.argmax(preds, 1)
        self.test_pyx_out.append(pyx)
        self.test_preds_out.append(preds)
        self.test_labels_out.append(test_labels.to(torch.int).cpu())
        self.test_img_paths_out.append(test_img_paths)

    def test_epoch_end(self, outputs):
        test_pyx_out = torch.cat(self.test_pyx_out)
        test_preds_out = torch.cat(self.test_preds_out)
        test_labels_out = torch.cat(self.test_labels_out)
        test_img_paths_out = sum(self.test_img_paths_out)

        test_probs = np.save("./Male_celeba_test_set_pyx", test_pyx_out.numpy())
        test_preds = np.save("./Male_celeba_pyx_argmax_predicted_labels", test_preds_out.numpy())
        test_labels = np.save("./Male_celeba_original_label", test_labels_out.numpy())
        test_img_paths = np.save("./Male_celeba_img_paths", np.array(test_img_paths_out))

    def configure_optimizers(self):
        scheduler = None
        optimizer = torch.optim.Adam(self.model.parameters())
        # "decay the learning rate with the cosine decay schedule without restarts"
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, args.epochs, eta_min=0, last_epoch=-1
        # )
        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}

# Configuration Variables
random_seed = 2
batch_size = 64
attribute = "Male"

reload = True
path_to_saved_models = "trained_models/"
model_filename = "epoch=03-Validation Loss_epoch=0.05803-Male_resnet50_fromScratch_CelebAAligned_5_epochs.ckpt"
hyperparamters = {}

train = False
val_only = False
test = True

save = True
saved_model_filename = f"{attribute}_resnet50_fromScratch_CelebAAligned_5_epochs"

epochs = 5
gpus = -1

wandb_logger = WandbLogger(name=saved_model_filename, project="cleanlab_celeba", entity='unr-mpl')

pl.seed_everything(random_seed)

if reload:
    model = Altered_Resnet.load_from_checkpoint(path_to_saved_models + model_filename, hyperparameters=hyperparamters)
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
        transform=None,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)

    val_dataset = CelebA_Dataset(
        split="validation",
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

if test:
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="auto",
        devices=1,
        num_nodes=1,
    )
else:
    trainer = pl.Trainer(
        logger=wandb_logger,
        # precision=16,
        callbacks=[checkpoint_callback],
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator="auto",
        devices=gpus,
        num_nodes=1,
        # limit_train_batches=0.5,
        # limit_val_batches=0.5,
        max_epochs=epochs
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