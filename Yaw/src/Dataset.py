import os
import torch
import cv2
from torch.utils.data import  Dataset, DataLoader
from torchvision.transforms import Compose,ToTensor,Resize


class Yaw(Dataset):
    def __init__(self,root ="data",train = True,transform = None):
        scipt_path = os.path.dirname(__file__)
        root = os.path.join(scipt_path,root)

        self.categories = ["Normal","Stroke"]
        self.list_img_name = []
        self.lable = []

        if train: data_path = os.path.join(root,"train")
        else: data_path = os.path.join(root,"test")

        for category in self.categories:
            category_path = os.path.join(data_path,category)
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path,img_name)
                self.list_img_name.append(img_path)
                self.lable.append(self.categories.index(category))

        self.transform = transform

    def __len__(self):
        return len(self.lable)

    def __getitem__(self, item):
        image = cv2.imread(self.list_img_name[item])
        image_path = self.list_img_name[item]
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        if self.transform:
            image = self.transform(image)
        lable = self.lable[item]
        return image, lable


if __name__ == '__main__':
    transform = Compose([ToTensor(),Resize((256,256))])
    dataset = Yaw(train=True,transform = transform)
    # dataset.__getitem__(709)
    dataloader = DataLoader(dataset=dataset,batch_size=8,shuffle=True,drop_last=True,num_workers=8)
    for img,lable in dataloader:
        print(img)
        print(lable)
    label_distribution = {}
    for images, labels in dataloader:
        print(images.shape)
        print(labels)
        for label in labels:
            label_distribution[label.item()] = label_distribution.get(label.item(), 0) + 1

    print("Label distribution across batches:", label_distribution)


