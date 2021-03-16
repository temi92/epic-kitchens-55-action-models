from pathlib import Path
import torch
import torchvision
import sys
sys.path.append("..")
from utils.transforms import GroupScale, GroupCenterCrop, GroupOverSample, Stack, ToTorchFormatTensor, GroupNormalize, GroupRandomHorizontalFlip, GroupMultiScaleCrop
from torchvision.transforms import Compose
from utils.dataset import DatasetWrapper, TSNDataSet, CustomDataSet
from gulpio.dataset import GulpImageDataset
from gulpio.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.hub
import torch.nn as nn
from trainer import Trainer
from models.model import Identity, get_tsn_model
import argparse

parser = argparse.ArgumentParser(description='Traning model using a transfer learning approach')
parser.add_argument("train_gulp", type=Path, help="path to training gulped data")
parser.add_argument("val_gulp", type=Path, help="path to val gulped data")
parser.add_argument("--epochs", default=10, type=int, help="number of total epochs to run")
parser.add_argument("--lr", "--learning_rate", default=0.001, type=float, help="learning rate")
parser.add_argument("--b", "--batch_size",default=16, type=int, help="batch size of data")

args = parser.parse_args()

#parse arguments
train_gulp = args.train_gulp
val_gulp = args.val_gulp
no_epochs = args.epochs
lr = args.lr
batch_size = args.b

config = {"lr": lr, "max_epochs": no_epochs} #config parameters for training model


#load model
base_model = "resnet50"
segment_count = 8
tsn = get_tsn_model(base_model=base_model, segment_count=segment_count, tune_model=True)


train_transform = Compose([
    GroupMultiScaleCrop(tsn.input_size),
    GroupRandomHorizontalFlip(), 
    Stack(roll=base_model == 'BNInception'),
    ToTorchFormatTensor(div=base_model != 'BNInception'),
    GroupNormalize(tsn.input_mean, tsn.input_std),
])


cropping = torchvision.transforms.Compose([
        GroupScale(tsn.scale_size),
        GroupCenterCrop(tsn.input_size),
    ])

test_transform = Compose([
    cropping,
    Stack(roll=base_model == 'BNInception'),
    ToTorchFormatTensor(div=base_model != 'BNInception'),
    GroupNormalize(tsn.input_mean, tsn.input_std),
])


print ("loading dataset...")

train_dataset = CustomDataSet("train_gulper/", transform=train_transform)
val_dataset = CustomDataSet("val_gulper/", transform=test_transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

#inputs, label = next(iter(dataloader))


#train model. 
trainer = Trainer(config, tsn, train_dataloader, val_dataloader)
trainer.train()
trainer.plot_results()




