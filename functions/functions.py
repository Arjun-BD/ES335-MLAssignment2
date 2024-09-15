import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.kernel_approximation import RBFSampler
from torchvision.transforms.functional import crop
import os
from sklearn import preprocessing

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import cv2


def create_coordinate_map(img, scale=1):
    num_channels, height, width = img.shape
    w_coords = torch.arange(0, width,  1/scale).repeat(int(height*scale), 1)
    h_coords = torch.arange(0, height, 1/scale).repeat(int(width*scale), 1).t()
    w_coords = w_coords.reshape(-1)
    h_coords = h_coords.reshape(-1)
    X = torch.stack([h_coords, w_coords], dim=1).float()
    X = X.to(device)
    Y = rearrange(img, 'c h w -> (h w) c').float()
    return X, Y

def load_and_preprocess_image(path, top = 500 , left = 800 , crop_size=400):
    img = torchvision.io.read_image(path)
    img_scaled = preprocessing.MinMaxScaler().fit_transform(img.reshape(-1, 1)).reshape(img.shape)
    img_scaled = torch.tensor(img_scaled)
    crop_img = crop(img_scaled, top , left , crop_size, crop_size)
    return crop_img

def scale_coordinates(Xcords):        
    scaler_X = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(Xcords.cpu())
    X_scaled = scaler_X.transform(Xcords.cpu())
    X_scaled = torch.tensor(X_scaled)
    X_scaled = X_scaled.float()
    return X_scaled, scaler_X

def create_rff_features(X, num_features, sigma, seed=42):
    rff = RBFSampler(n_components=num_features, gamma=1/(2 * sigma**2), random_state=seed)
    X_rff = torch.tensor(rff.fit_transform(X.cpu().numpy()), dtype=torch.float32).to(device)
    return X_rff

class LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

def train(net, lr, X, Y, epochs, verbose=True):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch} loss: {loss.item():.6f}")
    return loss.item()

def predict_image(net, X):
    net.eval()
    with torch.no_grad():
        return net(X)

def resize_image(img, size):
    img = rearrange(img, 'c h w -> h w c')
    resized = cv2.resize(img.cpu().numpy(), size, interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(rearrange(resized, 'h w c -> c h w'))

def plot_reconstructed_and_original_image(original_img, net, X, title=""):
    num_channels, height, width = original_img.shape
    net.eval()
    with torch.no_grad():
        outputs = net(X)
        outputs = outputs.reshape(height, width, num_channels)
    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax0.imshow(outputs.cpu())
    ax0.set_title("Reconstructed Image")
    ax1.imshow(original_img.cpu().permute(1, 2, 0))
    ax1.set_title("Original Image")
    for a in [ax0, ax1]:
        a.axis("off")
    fig.suptitle(title, y=0.9)
    plt.tight_layout()
    return outputs

def plot_images(original, downscaled, predicted, title=""):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(rearrange(original, 'c h w -> h w c').cpu().numpy())
    axs[0].set_title("Original")
    axs[1].imshow(rearrange(downscaled, 'c h w -> h w c').cpu().numpy())
    axs[1].set_title("Downscaled")
    axs[2].imshow(rearrange(predicted, 'c h w -> h w c').cpu().numpy())
    axs[2].set_title("Predicted")
    for ax in axs:
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def calculate_metrics(original, predicted):
    mse = nn.MSELoss()(original, predicted)
    rmse = torch.sqrt(mse)
    psnr = 20 * torch.log10(1 / rmse)
    return rmse.item(), psnr.item()


