# import all necessary packages
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path

# import relevant pytorch functions and packages
import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# import the model architecture
from model import MNISTClassifier_MLP

################################################
############### Load Real Data #################
################################################

# Load .png images and bring them into the correct format

def load_img(path):
    image = np.asarray(Image.open(path).convert("L"))
    pixels = np.array(image, copy=True)
    
    mask_255 = np.where(pixels == 255)
    mask_0 = np.where(pixels == 0)
    pixels[mask_255] = 0
    pixels[mask_0] = 255
    
    xr_image = xr.DataArray(
    pixels,
    dims=("y", "x"),
    name="brightness",
    )
    
    # image has dimensions ("y", "x")
    xr_image = xr_image.assign_coords(
        y=np.arange(xr_image.sizes["y"]),
        x=np.arange(xr_image.sizes["x"]),
    )

    target_y = np.linspace(xr_image.y.min().item(), xr_image.y.max().item(), 28)
    target_x = np.linspace(xr_image.x.min().item(), xr_image.x.max().item(), 28)

    resized = xr_image.interp(
        y=target_y,
        x=target_x,
        method="cubic",
    )

    return resized


# Now, we need a list of all images to load.
# We use os.listdir to list all files in the given directory:
init_path = Path.cwd() / "postal_code"

try:     
    list_of_pngs = os.listdir(init_path)
    
    # the loop itself
    imgs_mnist_size = []
    for png in list_of_pngs:
        if png.endswith(".png"):
            
            numpy_img = np.array(load_img(init_path / png))
            numpy_img = np.float32(numpy_img)
            
            # Convert 0–255 values to 0–1
            numpy_img /= numpy_img.max()
            tensor_img = torch.from_numpy(numpy_img)
            # Grayscale image: [28, 28] -> [1, 1, 28, 28]
            tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)
            
            imgs_mnist_size.append(tensor_img)
            
    print("Postal code images loaded")
except:
    print("The chosen path for your .png-files does not exist")

################################################
########## Load Real Trained Model #############
################################################

model = MNISTClassifier_MLP()  # same architecture/arguments used during training

device = "cpu"

state_dict = torch.load(
    Path.cwd() / "models" / "digit_prediction_model.pth",
    map_location=device,
    weights_only=True,
)

model.load_state_dict(state_dict)

################################################
########## Apply the Trained Model #############
################################################

img_number = 0
for dig in imgs_mnist_size:
    # use model on our resized and normalized .png images
    logits = model(dig)  
    predictions = logits.argmax(dim=1)
    print(f"Image {list_of_pngs[img_number]} is predicted to be {predictions[0]}")
    img_number += 1