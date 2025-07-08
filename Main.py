"""
Author: Georgios Voulgaris
Date: 07/07/2025
Description: The aim of this project is to apply Deep Learning techniques, to classify tree crown species from UAV
             images.
             The architectures under investigation are comprised of Convolutional Neural Networks, Transformer and
             Hybrid CNN-Transformer models. The dataset is comprised of UAV RGB and NDVI images depicting tree crowns of
             tree species from Wytham forest in Oxford.
             Moreover, conducting an investigation regarding Deep Learning Multimodal data fusion.
"""

# Imports
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
import wandb
import sys
import os
from utilities.Save import save_checkpoint, load_checkpoint
from utilities.Data import DataRetrieve
from utilities.Config import train_transforms, val_transforms, test_transforms
from utilities.Networks import networks
from utilities.Hyperparameters import arguments
from plots.ModelExam import parameters, get_predictions, plot_confusion_matrix, plot_most_incorrect, get_representations, get_pca, plot_representations, get_tsne


def step(data, targets, model, optimizer, criterion, train):
    with torch.set_grad_enabled(train):
        outputs = model(data)
        acc = outputs.argmax(dim=1).eq(targets).sum().item()
        loss = criterion(outputs, targets)

    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return acc, loss


@torch.no_grad()
def get_all_preds(model, loader, device):
    all_preds = []
    for x, _ in loader:
        x = x.to(device)
        preds = model(x)
        all_preds.append(preds)
    all_preds = torch.cat(all_preds, dim=0).cpu()
    return all_preds


def main():
    args = arguments()
    wandb.init(project="TreeCrownClass", config=args)

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")  # Can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # Load Data
    if args.dataset == 'TreeCrown512':
        dataset = ImageFolder("TreeCrown512")
    elif args.dataset == 'TreeCrown256':
        dataset = ImageFolder("TreeCrown256")
    elif args.dataset == 'TreeCrown128':
        dataset = ImageFolder("TreeCrown128")
    else:
        dataset = ImageFolder("TreeCrown64")
    print(f"Dataset is {args.dataset}")


if __name__ == "__main__":
    main()
    