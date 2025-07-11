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
from pandas import DataFrame
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
import wandb
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

    # Set Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")  # Can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # Load Data
    if args.dataset == 'TreeCrown_512':
        dataset = ImageFolder("TreeCrown_512")
    elif args.dataset == 'TreeCrown_256':
        dataset = ImageFolder("TreeCrown_256")
    elif args.dataset == 'TreeCrown_128':
        dataset = ImageFolder("TreeCrown_128")
    else:
        dataset = ImageFolder("TreeCrown_64")
    print(f"Dataset is {args.dataset}")

    labels = dataset.classes
    num_classes = len(labels)
    y = dataset.targets
    dataset_len = len(dataset)

    X_trainval, X_test, y_trainval, y_test = train_test_split(np.arange(dataset_len), y, test_size=0.2, stratify=y,
                                                              random_state=args.random_state, shuffle=True)
    X2 = X_trainval
    y2 = y_trainval
    X_train, X_val, y_train, y_val = train_test_split(X2, y2, test_size=0.2, stratify=y2,
                                                      random_state=args.random_state, shuffle=True)
    train_ds = Subset(dataset, X_train)
    val_ds = Subset(dataset, X_val)
    test_ds = Subset(dataset, X_test)
    print(f"len(test_ds): {len(test_ds)}")

    # Create train, validation and test datasets
    train_dataset = DataRetrieve(
        train_ds,
        transforms=train_transforms(args.width, args.height, args.Augmentation),
        augmentations=args.augmentation
    )

    val_dataset = DataRetrieve(
        val_ds,
        transforms=val_transforms(args.width, args.height)
    )

    test_dataset = DataRetrieve(
        test_ds,
        transforms=test_transforms(args.width, args.height)
    )
    print(f"len(test_dataset): {len(test_dataset)}")

    # Create train, validation and test dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    prediction_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    print(f"len(prediction_loader.dataset): {len(prediction_loader.dataset)}")

    # Network
    model = networks(architecture=args.architecture, in_channels=args.in_channels, num_classes=num_classes,
                     pretrained=args.pretrained, requires_grad=args.requires_grad,
                     global_pooling=args.global_pooling).to(device)
    print(model)
    n_parameters = parameters(model)
    print(f"The model has {n_parameters:,} trainable parameters")

    # Loss and Optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load Model
    if args.load_model == 'True':
        print(f"Load model is {args.load_model}")
        if device == torch.device("cpu"):
            load_checkpoint(torch.load("my_checkpoint.pth.tar", map_location=torch.device('cpu')), model, optimizer)
        else:
            load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    # Train Network
    for epoch in range(args.epochs):
        model.train()
        sum_acc = 0
        for data, targets in train_loader:
            data = data.to(device=device)
            targets = targets.to(device=device)

            if args.augmentation == "cutmix":
                None
            else:
                acc, loss = step(data, targets, model=model, optimizer=optimizer, criterion=criterion, train=True)
                sum_acc += acc

        train_avg_acc = sum_acc / len(train_loader)

        # Saving model
        if args.save_model == 'True':
            if epoch % 10 == 0:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint)

        # Evaluate Network
        model.eval()
        sum_acc = 0
        for data, targets in val_loader:
            data = data.to(device=device)
            targets = targets.to(device=device)
            val_acc, val_loss = step(data, targets, model=model, optimizer=optimizer, criterion=criterion,
                                     train=False)
            sum_acc += val_acc
        val_avg_acc = sum_acc / len(val_loader)

        print(
            f"Epoch: {epoch + 1} \tTraining accuracy: {train_avg_acc:.2f} \n\t\tValidation accuracy: {val_avg_acc:.2f}")

        train_steps = len(train_loader) * (epoch + 1)
        wandb.log({"Train Accuracy": train_avg_acc, "Validation Accuracy": val_avg_acc}, step=train_steps)

    train_preds = get_all_preds(model, loader=prediction_loader, device=device)
    print(f"Train predictions shape: {train_preds.shape}")
    print(f"The label the network predicts strongly: {train_preds.argmax(dim=1)}")
    predictions = train_preds.argmax(dim=1)

    # Most Confident Incorrect Predictions
    images, labels, probs = get_predictions(model, prediction_loader, device)

    pred_labels = torch.argmax(probs, 1)
    print(f"pred_labels: {pred_labels}")

    corrects = torch.eq(labels, pred_labels)

    incorrect_examples = []

    for image, label, prob, correct in zip(images, labels, probs, corrects):
        if not correct:
            incorrect_examples.append((image, label, prob))
    incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)

    # n_images = 48
    n_images = min(48, len(incorrect_examples))
    classes = dataset.classes
    """if args.dataset == 'TreeCrown_512':
        classes = os.listdir('TreeCrown_512')
    elif args.dataset == 'TreeCrown_256':
        classes = os.listdir('TreeCrown_256')
    elif args.dataset == 'TreeCrown_128':
        classes = os.listdir('TreeCrown_128')
    else:
        classes = os.listdir('TreeCrown_64')"""

    print(f"dataset.classes: {dataset.classes}")
    print(f"classes used in plot: {classes}")

    plot_most_incorrect(incorrect_examples, classes, n_images)
    wandb.save('Most_Conf_Incorrect_Pred.png')

    # Principle Components Analysis (PCA)
    outputs, intermediates, labels = get_representations(model, train_loader, device)

    output_pca_data = get_pca(outputs)
    plot_representations(output_pca_data, labels, classes, "PCA")
    wandb.save('PCA.png')

    # Intermediate Principle Components Analysis (PCA)
    intermediate_pca_data = get_pca(intermediates)
    plot_representations(intermediate_pca_data, labels, classes, "INTPCA")
    wandb.save('Intermediate_PCA.png')

    # t-Distributed Stochastic Neighbor Embedding (t-SNE)
    n_images = 10_000

    output_tsne_data = get_tsne(outputs, n_images=n_images)
    plot_representations(output_tsne_data, labels, classes, "TSNE", n_images=n_images)
    wandb.save('TSNE.png')

    # Intermediate t-Distributed Stochastic Neighbor Embedding (t-SNE)
    intermediate_tsne_data = get_tsne(intermediates, n_images=n_images)
    plot_representations(intermediate_tsne_data, labels, classes, "INTTSNE", n_images=n_images)
    wandb.save('Intermediate_TSNE.png')

    print(f"Classes (class names): {classes}")
    print(f"Labels (true labels y_test): {y_test[:10]}")  # print first 10 for brevity

    print(f"y_test shape: {len(y_test)}")
    print(f"predictions shape: {train_preds.argmax(dim=1).shape}")
    print(f"First 10 y_test: {y_test[:10]}")
    print(f"First 10 predictions: {train_preds.argmax(dim=1)[:10]}")

    plot_confusion_matrix(y_test, train_preds.argmax(dim=1), classes)
    wandb.save('Confusion_Matrix.png')

    # Confusion Matrix
    wandb.sklearn.plot_confusion_matrix(y_test, train_preds.argmax(dim=1), labels)

    # Class proportions
    wandb.sklearn.plot_class_proportions(y_train, y_test, labels)
    precision, recall, f1_score, support = score(y_test, train_preds.argmax(dim=1))
    test_acc = accuracy_score(y_test, train_preds.argmax(dim=1))
    wandb.log({"Test Accuracy": test_acc})

    print(f"Test Accuracy: {test_acc}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1_score: {f1_score}")
    print(f"support: {support}")

    # Use classes as dataset.classes or the label list from y_test's unique classes
    unique_labels = np.unique(y_test)

    # If I want class names, map unique_labels (indices) to class names
    class_names = [dataset.classes[i] for i in unique_labels]

    # Test data saved in Excel document
    df = DataFrame({'class': class_names,  # labels,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'support': support})

    # Append a final row for accuracy
    df.loc[len(df.index)] = ['Overall Accuracy', '', '', '', test_acc]

    df.to_excel('test.xlsx', sheet_name='sheet1', index=False)
    df.to_csv('test.csv', index=False)
    compression_opts = dict(method='zip', archive_name='out.csv')
    df.to_csv('out.zip', index=False, compression=compression_opts)

    wandb.save('test.csv')
    wandb.save('my_checkpoint.pth.tar')
    wandb.save('Predictions.csv')


if __name__ == "__main__":
    main()
