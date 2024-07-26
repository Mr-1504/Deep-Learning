#resnet
import sys
sys.path.append('/content/drive/MyDrive/Train')
import os
import shutil
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from Dataset import Yaw
from torchvision.transforms import Compose, ToTensor, Resize, RandomRotation, RandomHorizontalFlip, ColorJitter, GaussianBlur
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm.autonotebook import tqdm
import warnings
import torchvision.models as models
import random
warnings.filterwarnings("ignore")
import cv2

def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap="plasma")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Stroke'])
    cm_display.plot()
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("\n\nMetrics:")
    print("Accuracy: ", round(accuracy, 2))
    print("Precision: ", round(precision, 2))
    print("Recall: ", round(recall, 2))
    print("F1-score: ", round(f1_score, 2))

def get_args():
    parser = argparse.ArgumentParser(description='Classifier')
    parser.add_argument('-p', '--data_path', type=str, default="_data_")
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-l', '--lr', type=float, default=1e-3)
    parser.add_argument('-s', '--image_size', type=int, default=224)
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None)
    parser.add_argument('-t', '--tensorboard_path', type=str, default="/content/drive/MyDrive/Train/tensorboard")
    parser.add_argument('-r', '--trained_path', type=str, default="/content/drive/MyDrive/Train/trained_models")
    args, unknown = parser.parse_known_args()
    return args

def train(args, train_accuracies_epochs, train_loss_value_epochs, val_accuracies_epochs, val_loss_value_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
        RandomRotation(degrees=15),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
    ])
    train_set = Yaw(root=args.data_path, train=True, transform=transform)
    valid_set = Yaw(root=args.data_path, train=False, transform=transform)

    sample_images = []
    for i in range(10):
        random_integer = random.randint(0, len(train_set)-1)
        image, _ = train_set[random_integer]
        sample_images.append(image)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for ax, img in zip(axes, sample_images):
        arrImg = img.numpy().transpose(1, 2, 0)
        cvImg = cv2.cvtColor(arrImg, cv2.COLOR_RGB2BGR)
        ax.imshow(cvImg)
    plt.tight_layout()
    plt.show()

    training_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": 6
    }

    valid_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": 6
    }
    train_dataloader = DataLoader(train_set, **training_params)
    valid_dataloader = DataLoader(valid_set, **valid_params)

    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
    else:
        start_epoch = 0
        best_acc = 0

    if os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)
    os.mkdir(args.tensorboard_path)
    if not os.path.isdir(args.trained_path):
        os.mkdir(args.trained_path)
    writer = SummaryWriter(args.tensorboard_path)
    num_iters = len(train_dataloader)

    conf_matrix = None
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        loss_value_iter = []
        correct_sample = 0
        total_sample = 0

        progress_bar = tqdm(train_dataloader, colour="yellow")
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()

            _, predicted = torch.max(predictions.data, 1)
            total_sample += labels.size(0)
            correct_sample += (predicted == labels).sum().item()

            progress_bar.set_description("Epoch {}/{}. Loss value: {:.4f}".format(epoch + 1, args.epochs, loss_value))
            losses.append(loss_value)
            writer.add_scalar("Train/Loss", np.mean(losses), epoch*num_iters+iter)
            loss_value_iter.append(loss_value)

        epoch_accuracy = correct_sample / total_sample
        train_accuracies_epochs.append(epoch_accuracy)
        train_loss_value_epochs.append(np.mean(loss_value_iter))

        model.eval()
        losses = []
        all_predictions = []
        all_gts = []
        val_loss_value_iter = []
        with torch.no_grad():
            for iter, (images, labels) in enumerate(valid_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                predictions = model(images)
                max_idx = torch.argmax(predictions, 1)
                loss = criterion(predictions, labels)
                losses.append(loss.item())
                all_gts.extend(labels.tolist())
                all_predictions.extend(max_idx.tolist())
                val_loss_value_iter.append(loss.item())
        val_loss_value_epochs.append(np.mean(val_loss_value_iter))
        writer.add_scalar("Val/Loss", np.mean(losses), epoch)
        acc = accuracy_score(all_gts, all_predictions)
        val_accuracies_epochs.append(acc)
        writer.add_scalar("Val/Accuracy", acc, epoch)
        conf_matrix = confusion_matrix(all_gts, all_predictions)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "batch_size": args.batch_size
        }

        torch.save(checkpoint, os.path.join(args.trained_path, "last.pt"))
        if acc > best_acc:
            torch.save(checkpoint, os.path.join(args.trained_path, "best.pt"))
            best_acc = acc

        scheduler.step()
    plot_confusion_matrix(writer, conf_matrix, [i for i in range(len(train_set.categories))], epoch)

def plot_phase(train_accuracies_epochs, train_loss_value_epochs, val_accuracies_epochs, val_loss_value_epochs):
    epochs = range(1, len(train_accuracies_epochs) + 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies_epochs, color='blue', label='Train')
    plt.plot(epochs, val_accuracies_epochs, color='orange', linestyle='--', label='Test')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss_value_epochs, color='blue', label='Training Loss')
    plt.plot(epochs, val_loss_value_epochs, color='orange', linestyle='--', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/Train/ResNet.png')

if __name__ == "__main__":
    args = get_args()
    train_accuracies_epochs = []
    train_loss_value_epochs = []
    val_accuracies_epochs = []
    val_loss_value_epochs = []
    train(args, train_accuracies_epochs, train_loss_value_epochs, val_accuracies_epochs, val_loss_value_epochs)
    plot_phase(train_accuracies_epochs, train_loss_value_epochs, val_accuracies_epochs, val_loss_value_epochs)
