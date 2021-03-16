from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose
import sys
sys.path.append("..")
from models.model import get_tsn_model, get_trn_model
from utils.transforms import GroupScale, GroupCenterCrop, GroupOverSample, Stack, ToTorchFormatTensor, GroupNormalize

from utils.dataset import DatasetWrapper, TSNDataSet
from utils.metrics import accuracy
from utils.plot_results import create_bar_plot
import torch.nn.functional as F
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
parser = argparse.ArgumentParser(description="comparison bwteen tsn and trn models with resnet50 backbone")
parser.add_argument("gulped_dir", type=Path, help="path to gulped file")
args = parser.parse_args()
gulp_dir = args.gulped_dir

segment_count = 8
base_model = 'resnet50'

tsn = get_tsn_model(base_model=base_model, segment_count=segment_count, tune_model=False)
trn = get_trn_model(base_model=base_model, segment_count=segment_count, tune_model=False)

### evaluation mode
tsn.eval()
trn.eval()
cropping = torchvision.transforms.Compose([
        GroupScale(tsn.scale_size),
        GroupCenterCrop(tsn.input_size),
    ])

transform = Compose([
    cropping,
    Stack(roll=base_model == 'BNInception'),
    ToTorchFormatTensor(div=base_model != 'BNInception'),
    GroupNormalize(tsn.input_mean, tsn.input_std),
])


batch_size = 1
height, width = 224, 224
dataset = DatasetWrapper(gulp_dir)

dataset = TSNDataSet(dataset, test_mode=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)



def compute_stats(model, data, label):
    """
    performs inference on the data set and returns the loss, acc 
    """
    start_time = time.time()
    verb_logits, noun_logits  = model(data)
    fps = 1.0/((time.time() - start_time)/data.shape[1])
   
 
    outputs = {"verb":verb_logits, "noun":noun_logits}

    tasks = {
            task: {
                "output": outputs[task],
                "preds": outputs[task].argmax(-1),
                "labels": label[f"{task}_class"],
                "weight": 1,
            }
            for task in ["verb", "noun"]
        }
    results = dict()
    loss = 0
  
    for task, d in tasks.items():
        task_loss = F.cross_entropy(d["output"], d["labels"])
        loss += d["weight"] * task_loss
        accuracy_1, accuracy_5 = accuracy(d["output"], d["labels"], ks=(1, 5))

        #compute accuracy ...
        results[f"{task}_accuracy@1"] = accuracy_1
        results[f"{task}_accuracy@5"] = accuracy_5

        #compute loss
        results[f"{task}_loss"] = task_loss
        results[f"{task}_preds"] = d["preds"]
    results["video_id"] = label["video_id"]
    results["loss"] = loss/len(tasks)
    results["fps"] = fps
    return results

models_verb_acc = []
models_noun_acc = []

models_verb_loss = []
models_noun_loss = []

models_fps = []

for model in [tsn, trn]:
    verb_acc = []
    noun_acc = []
    fps = []

    verb_loss = []
    noun_loss = []  
    for i, (data, label) in enumerate(dataloader):
 

        ## just analysing certain section of the data to save computational time
        if i  > 1:
            break   
        data = data.reshape((batch_size, -1, height, width))
        results = compute_stats(model, data, label)
        verb_acc.append(results["verb_accuracy@5"])
        noun_acc.append(results["noun_accuracy@5"])
        fps.append(results["fps"])

        verb_loss.append(results["verb_loss"])
        noun_loss.append(results["noun_loss"])
    
    fps_avg = sum(fps)/len(fps)
    models_fps.append(fps_avg)

    verb_acc_avg = torch.mean(torch.stack(verb_acc))
    verb_loss_avg = torch.mean(torch.stack(verb_loss))

    noun_acc_avg = torch.mean(torch.stack(noun_acc))
    noun_loss_avg = torch.mean(torch.stack(noun_loss))

    models_verb_acc.append(verb_acc_avg.cpu().detach().numpy())
    models_noun_acc.append(noun_acc_avg.cpu().detach().numpy())

    models_verb_loss.append(verb_loss_avg.cpu().detach().numpy())
    models_noun_loss.append(noun_loss_avg.cpu().detach().numpy())


create_bar_plot(models_noun_acc, models_verb_acc, models_noun_loss, models_verb_loss, models_fps)

