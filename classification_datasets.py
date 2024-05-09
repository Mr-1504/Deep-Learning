import os
import cv2
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomAffine, ColorJitter
from PIL import Image


class CIFARDataset(Dataset):
    def __init__(self, root="data", train=True, transform=None):
        data_path = os.path.join(root, "cifar-10-batches-py")
        if train:
            data_files = [os.path.join(data_path, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(data_path, "test_batch")]
        self.images = []
        self.labels = []
        for data_file in data_files:
            with open(data_file, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
                self.images.extend(data[b'data'])
                self.labels.extend(data[b'labels'])
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = self.images[item].reshape((3, 32, 32)).astype(np.float32)
        if self.transform:
            image = np.transpose(image, (1, 2, 0))
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)
        label = self.labels[item]
        return image, label

class AnimalDataset(Dataset):
    def __init__(self, root="data/animals", train=True, transform=None):
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
        if train:
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")

        self.image_paths = []
        self.labels = []

        for category in self.categories:
            category_path = os.path.join(data_path, category)
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                self.image_paths.append(image_path)
                self.labels.append(self.categories.index(category))

        self.transform = transform


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        if transform:
            image = self.transform(image)
        label = self.labels[item]
        return image, label


if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])

    dataset = AnimalDataset(root="./data/animals", train=True, transform=transform)
    image, label = dataset.__getitem__(12347)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=6)
    for images, labels in dataloader:
        print(images.shape)
        print(labels)
