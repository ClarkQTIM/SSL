# Imports

## Standard
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import json

## DL
import torch
from torch.nn import CrossEntropyLoss
from torch import nn

# Training

## Cross Entropy with Logits
class Train:

    def __init__(self, model, W, epochs, train_dataloader, learning_rate, model_directory, title, results_saving, plotting):
        self.model = model
        self.W = W
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.learning_rate = learning_rate
        self.model_directory = model_directory
        self.title = title
        self.results_saving = results_saving
        self.plotting = plotting

    def configure_devices(self, device_ids):
        ''' Method to enforce parallelism during model training '''
        if  len(device_ids) > 1: #isinstance(device_ids, list): # for multiple GPU training
            print(f'Using multiple GPUS: {device_ids}')
            self.base_device = 'cuda:{}'.format(device_ids[0])
            self.model.to(self.base_device)
            self.model = nn.DataParallel(self.network, device_ids=device_ids)
            print(f'Base device is {self.base_device}')
        elif len(device_ids) == 1 and device_ids[0].isdigit(): # for single GPU training
            print(f'Using GPU ', device_ids[0])
            self.base_device = 'cuda:' + device_ids[0]
            self.model = self.model.to(self.base_device)
            self.W = self.W.to(self.base_device)
            print(f'Base device is {self.base_device}')
        else:
            print('Using CPU')
            self.base_device = 'cpu' # for CPU based training
            self.model = self.model.to('cpu')
            print(f'Base device is {self.base_device}')

    def fit(self):

        # Loss Function, Optimizer, and Scheduler
        warmup_epochs = 10
        optimizer = torch.optim.AdamW(self.model.parameters(), self.learning_rate, weight_decay = 0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(self.train_dataloader) * (self.epochs- warmup_epochs))
        loss_function = CrossEntropyLoss(label_smoothing = 0.8)

        start = time.time()

        # Lists into which we will populate our metrics
        train_loss_values = list()
        train_top1_values = list()
        train_top5_values = list()

        # Training and Validation Loop
        for epoch in range(self.epochs):

            if epoch in range(0, warmup_epochs):
                optimizer_w = torch.optim.AdamW(self.model.parameters(), 0+0.01*epoch, weight_decay = 0.05)

            else:
                print('We are out of the warm-up phase and revert to our original optimizer and scheduler')

            ## Epoch tracking
            print("-" * 10)
            print(f"epoch {epoch + 1}/{self.epochs}")

            self.model.train() # Making sure we are in train mode, as we have both dropout and batch normalization we want running
            
            ## Initializing epoch loss and step, as we will have find the average of loss versus steps
            train_epoch_loss = 0
            step = 0
            top1_value = 0
            top5_value = 0

            ## Stepping through the batches
            for batch_data in self.train_dataloader: # Ranging over the batches
                step += 1 # We are calculating the loss per step and then will find the average at the end of the epoch
                inputs, labels = batch_data[0].to(self.base_device), batch_data[1].to(self.base_device) # Getting input images and their segmentations
                if epoch in range(0, warmup_epochs): # Zeroing out the optimizer gradients
                    optimizer_w.zero_grad() 
                else:
                    optimizer.zero_grad()
                model_outputs = self.model(inputs) # Getting the model outputs
                class_outputs = self.W(model_outputs) # Getting the class probabilties
                loss = loss_function(class_outputs, labels) # Calculating the loss
                loss.backward() # Updating the parameters
                if epoch in range(0, warmup_epochs): # Updating the optimizer
                    optimizer_w.step()
                else:
                    optimizer.step() 
                train_epoch_loss += loss.item() # Getting the loss and adding it to the total loss for this epoch

                ### Top 1 and 5 Accuracy
                probs, pred_classes = class_outputs.topk(5, 1, largest=True, sorted=True) # Getting the top 5 classes based on probability. Size (batch_size, num_top)
                labels_reshaped = labels.view(labels.size(0), -1).expand_as(pred_classes) # Reshaping our test_labels from (batch_size,) t0 (batch_size, num_top)
                correct = pred_classes.eq(labels_reshaped).float()
                correct_5 = correct[:, :5].sum()/len(inputs) # Percentage of predictions in the top 5 for this batch
                correct_1 = correct[:, :1].sum()/len(inputs) # Percentage of predictions in the top 1 for this batch
                top1_value += correct_1.item()
                top5_value += correct_5.item()

            ## Scheduler stepping
            if epoch in range(0, warmup_epochs):
                print('No scheduler during warmup')
                print('Learning rate during warmup phase:',optimizer_w.param_groups[0]['lr'])
            else:
                scheduler.step() # Stepping the learning rate scheduler
            
            ## Epoch loss, top1, and top5 calculation and printing
            train_epoch_loss /= step
            top1_value /= step
            top5_value /= step
            train_loss_values.append(train_epoch_loss)
            train_top1_values.append(top1_value)
            train_top5_values.append(top5_value)

            print(f"epoch {epoch + 1} average loss: {train_epoch_loss:.4f}")
            print(f"epoch {epoch + 1} average Top1: {top1_value:.4f}")
            print(f"epoch {epoch + 1} average Top5: {top5_value:.4f}")

            ## Validation/Testing doesn't really make sense here, so we drop it

            ## Plotting
            if self.plotting:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                ax1.plot(train_loss_values)
                ax1.legend(['Epoch Loss'])
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Loss Performance')    
                ax2.plot(train_top1_values)
                ax2.plot(train_top5_values)
                ax2.legend(['Train Top1', 'Train Top5'])
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Percent')
                ax2.set_title('Top1 and Top5 Performance')     
                plt.savefig(self.model_directory + self.title + 'performance.png')
                plt.close()

            ## Model weights saving
            if self.results_saving:
                ### Losses
                # np.save(os.path.join(self.model_directory, self.title + "epoch_loss.npy"), epoch_)
                ### Weight saving
                    if train_loss_values[-1] == np.min(train_loss_values): # Checking if this epoch did better than the last on validation. If so, we save it.
                        savepath = os.path.join('Best_Models/', self.title + "best_epoch.pth")
                        print("savepath: ", savepath)
                        torch.save(self.model.state_dict(), savepath)
                        print("saved model")
                    else:
                        print(f'Epoch {epoch+1} did not have as high performance as the previous, so we will not save these weights.')

        end = time.time()
        print('This training took', round(end-start, 3), 'seconds.')     