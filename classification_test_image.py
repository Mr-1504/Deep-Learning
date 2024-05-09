import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
import warnings
from Model import CNN
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description="Animal classifier")
    parser.add_argument("-s", "--size", type=int, default=224)
    parser.add_argument("-i", "--image_path", type=str, default="test_images/1.jpg")
    parser.add_argument("-c", "--checkpoint_path", type=str, default="trained_models/best.pt")
    args = parser.parse_args()
    return args


def test(args):
    categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=len(categories)).to(device)
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        model.eval() #khai báo vào quá trình test
    else:
        print("A checkpoint must be provided")
        exit(0)
    if not args.image_path:
        print("An image must be provided")
        exit(0)
    image = cv2.imread(args.image_path)
    image = cv2.resize(image, (args.size, args.size))
    image = np.transpose(image, (2, 0, 1))[None, :, :, :]
    image = image / 255
    # image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).to(device).float()
    softmax = nn.Softmax()
    with torch.no_grad():
        prediction = model(image)
    probs = softmax(prediction)
    max_value, max_index = torch.max(probs, dim=1)
    print("This image is about {} with probability of {}".format(categories[max_index], max_value[0].item()))
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(categories, probs[0].cpu().numpy())
    ax.set_xlabel("Animal")
    ax.set_ylabel("Probability")
    ax.set_title(categories[max_index])
    plt.savefig("animal_prediction.png")



if __name__ == "__main__":
    args = get_args()
    test(args)
