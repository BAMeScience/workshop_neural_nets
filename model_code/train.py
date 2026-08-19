# import all necessary packages
import numpy as np
import pandas as pd
from pathlib import Path

# import relevant pytorch functions and packages
import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from model import MNISTClassifier_MLP

################################################
################### The Data ###################
################################################

# Download MNIST train and validation data 
train_data = MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

print("Training data loaded")

val_data = MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

print("Validation data loaded")

# Create the dataloader objects for easy batch handling
train_loader = DataLoader(
    train_data,
    batch_size=64,   
    shuffle=True,    
)

val_loader = DataLoader(
    val_data,
    batch_size=64,
    shuffle=False, 
)

print("Dataloader created")

################################################
#### Define all model dependent variables ######
################################################

# the model 
model = MNISTClassifier_MLP()

# the loss
loss_function = nn.CrossEntropyLoss()

# the optimzer with learning rate 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# the number of epochs
epochs = 5

################################################
############# The actual training ##############
################################################

for epoch in range(epochs):
    
    print(f"Start Epoch {epoch+1}")
    
    # Training
    # turn on training mode
    model.train()            
    # initialize loss for measuring its development over the epochs           
    total_loss = 0.0                    

    for images, labels in train_loader: # iterate over every batch in the train_loader and retrieve images and labels 
        # Set the gradients from the previous batch to 0 before calculating gradients for the current batch
        optimizer.zero_grad()           
        # Call the model. Input: images (see forward function of model); Output: Vector with 10 entries
        logits = model(images)          
        # Call the loss function. nn.CrossEntropyLoss() takes predictions first, true values second.
        loss = loss_function(logits,labels)    
        # Calculate the gradients via backpropagation                     
        loss.backward()     
        # Update the models training parameters according to the gradients            
        optimizer.step()                
        # Update the loss with the loss from this batch. Displaying purposes only
        total_loss += loss.item()       

    # Average the loss for this epoch
    average_loss = total_loss / len(train_loader) 

    ################################################################################################
    ################################################################################################
    
     # Evaluation
    model.eval() # turn on evaluation mode
    correct = 0  # track the number of correctly assigned digits
    total = 0    # # track the total number of samples 
    validation_loss = 0.0 

    with torch.no_grad():
        for images, labels in val_loader:
            
            images = images                                 # Maybe add the .device()
            labels = labels

            logits = model(images)                          # run the trained model on the validation data
            
            predictions = logits.argmax(dim=1)              

            loss = loss_function(logits,labels)
            validation_loss += loss.item()  
                        
            correct += (predictions == labels).sum().item() # Save correctly predicted digits of the batch 
            total += labels.size(0)                         # Save total size of the batch

    # Calculate average loss per epoch for validation samples
    average_val_loss = validation_loss / len(val_loader)
    # Calculate the proportion of correctly predicted digits 
    accuracy = correct / total     
    
    # Print the loss of each epoch. We want them to go down
    print(f"Epoch {epoch + 1}/{epochs}, training loss: {average_loss:.4f}, validation loss: {average_val_loss:.4f}, validation accuracy: {accuracy:.2%}") 
    
    
# After training, we save the final model for later usage

# get current working directory
cwd = Path.cwd()
# save it in directorey
path_to_save_in = cwd / "models" 

if not path_to_save_in.exists():
    path_to_save_in.mkdir(parents=True)

torch.save(model.state_dict(), path_to_save_in / "digit_prediction_model.pth")
