#########
# Imports
#########

# Standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time

# DL
import torch
import torch.nn as nn
from monai.data import DataLoader

# Py Files
from utils import *

##########
# Training
##########

class Train:
    # Initialization
    def __init__(self, model, base_device, feature_extractor, epochs, train_dataloader, val_dataloader, learning_rate, mask_ratio, 
                 visualization, model_directory, title, seed):
        self.model = model
        self.feature_extractor = feature_extractor
        self.base_device = base_device
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.mask_ratio = mask_ratio
        self.visualization = visualization
        self.model_directory = model_directory
        self.title = title
        self.seed = seed

    def fit(self):
        # Training and validation loop

        start = time.time()

        val_interval = 1

        train_loss_values = []
        val_loss_values = []

        optimizer = torch.optim.AdamW(self.model.parameters(), self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=self.epochs)

        for epoch in range(self.epochs):

                ## Epoch tracking
                print("-" * 10)
                print(f"epoch {epoch + 1}/{self.epochs}")

                self.model.train() # Making sure we are in train mode, as we have both dropout and batch normalization we want running

                ## Initializing epoch loss and step, as we will have to find the average of loss versus steps
                train_loss = 0 # Epoch train loss
                step = 0

                ## Stepping through the batches
                for batch_data in self.train_dataloader: # Ranging over the batches
                    step += 1 # Counting the batches
                    input = batch_data
                    optimizer.zero_grad() # Zeroing out the optimizer gradients

                    if self.mask_ratio != None:
                        loss, y, mask = self.model(input.to(self.base_device), mask_ratio=self.mask_ratio)
                    else:
                        output = self.model(input.to(self.base_device))
                        loss = output['loss']
                        y = output['logits']
                        mask = output['mask']

                    loss.backward() # Updating the parameters
                    optimizer.step() # Updating the optimizer

                    train_loss += loss.item() # Getting the loss and adding it to the total loss for this epoch
                
                ## Scheduler step at end of epoch
                scheduler.step()

                ## Epoch loss calculation and printing
                train_loss /= step

                train_loss_values.append(train_loss)
                print(f'Train Epoch Loss {np.round(train_loss, 4)}')

                ## Validation
                if (epoch + 1) % val_interval == 0:

                    self.model.eval() # Turning off the random components (dropout and batch norm) for evaluation
                    ### Stepping through the validation batch
                    with torch.no_grad(): # We are not updating any gradients during this
                        val_loss = 0 # Epoch validation loss
                        val_step = 0

                        for batch_data in self.val_dataloader: # Running through the validation dataset batches
                            val_step += 1 # Counting the batches
                            val_input = batch_data

                            if self.mask_ratio != None:
                                loss, y, mask = self.model(val_input.to(self.base_device), mask_ratio=self.mask_ratio)
                            else:
                                output = self.model(val_input.to(self.base_device))
                                loss = output['loss']
                                y = output['logits']
                                mask = output['mask']
                            
                            val_loss += loss.item() # Adding the val loss for this epoch to the running total

                        val_loss /= val_step
                        val_loss_values.append(val_loss) # Adding the average loss for this epoch validation iteration
                        
                    ### Printing our performance this validation iteration
                    print(f'Val Epoch Loss {np.round(val_loss, 4)}')

                ## Plotting
                plt.figure(figsize=(10, 10))
                plt.plot(train_loss_values)
                plt.plot(val_loss_values)
                plt.legend(['Training Loss', 'Validation Loss'])
                plt.title('Training Performance')    
                plt.xlabel('Epoch')
                plt.ylabel('Loss')     
                plt.savefig(self.model_directory + self.title + '_performance.png')
                plt.close()

                ## Model weights saving
                ### Weight saving
                if val_loss_values[-1] == np.min(val_loss_values): # Checking if this epoch has the lowest validation. If so, we save it.
                    savepath = '/sddata/projects/SSL/custom_mae/Reconstruction_Custom_Finetuning_Best_Models/' + self.title + "_best_epoch.pth"
                    print("savepath: ", savepath)
                    torch.save(self.model.state_dict(), savepath)
                    print("saved model")
                else:
                    print(f'Epoch {epoch+1} did not have as high performance as the previous, so we will not save these weights.')
                        
                ### Model metrics saving
                np.save(os.path.join(self.model_directory, self.title + "_epoch_loss.npy"), train_loss_values)
                np.save(os.path.join(self.model_directory, self.title + "_val_loss.npy"), val_loss_values)

                if (epoch+1) % self.visualization == 0:

                    visualize(epoch, val_input[0].unsqueeze(0).to(self.base_device), self.model, self.feature_extractor, False, True, self.model_directory, self.title)

        end = time.time()

        print(f'We are done! This training took: {end-start} seconds.')
