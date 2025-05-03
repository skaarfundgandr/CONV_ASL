import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import pandas as pd

import utils
# Dataset
train_df = pd.read_csv("data/train.csv")
valid_df = pd.read_csv("data/valid.csv")
# Model and input params
IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHS = 1
N_CLASSES = 24
# TODO: Cleanup
class MyDataset(Dataset):
    def __init__(self, base_df):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_df = base_df.copy()
        y_df = x_df.pop("label")
        x_df = x_df.values / 255
        x_df = x_df.reshape(-1, IMG_CHS, IMG_WIDTH, IMG_HEIGHT)
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).to(device)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    def __len__(self):
        return len(self.xs)
# Trains and exports the model
def train_model(export=True, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = 32 # Number of batches
    train_data = MyDataset(train_df)
    train_loader = DataLoader(train_data, batch_size=n, shuffle=True)
    train_N = len(train_loader.dataset)

    valid_data = MyDataset(valid_df)
    valid_loader = DataLoader(valid_data, batch_size=n)
    valid_N = len(valid_loader.dataset)

    flattened_img_size = 75 * 3 * 3

    base_model = nn.Sequential(
        utils.ConvNN(IMG_CHS, 25, 0), # 25 x 14 x 14
        utils.ConvNN(25, 50, 0.2), # 50 x 7 x 7
        utils.ConvNN(50, 75, 0),  # 75 x 3 x 3
        # Flatten to Dense Layers
        nn.Flatten(),
        nn.Linear(flattened_img_size, 512),
        nn.Dropout(.3),
        nn.ReLU(),
        nn.Linear(512, N_CLASSES)
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(base_model.parameters())

    model = torch.compile(base_model.to(device))
    # Data Augmentation:
    # Add random rotations, Crops, Flips, and Color Jitter to data
    random_transforms = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(.9, 1), ratio=(1, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.2, contrast=.5)
    ])

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        utils.train(model, train_loader, train_N, random_transforms, optimizer, loss_function)
        utils.validate(model, valid_loader, valid_N, loss_function)

    if export:
        torch.save(base_model, 'model.pth')

    return model