#########
# Imports
#########

# Standard
import os
import numpy as np
import pandas as pd
from PIL import Image as PImage # We are loading this in as PImage because we will also load in something called image from datasets
import matplotlib.pyplot as plt
import random
import PIL
import concurrent.futures
import sys

# DL
import torch
import torch.nn as nn
import transformers
from transformers import ViTFeatureExtractor, ViTMAEForPreTraining, ViTForImageClassification
from torch.utils.data import Dataset
from datasets import DatasetDict, Image
from torch.utils.data import random_split

# Python Scripts
import models_mae

# Image manipulation
from PIL import Image, ImageOps, ImageFile

###########
# Functions
###########

# Configure Devices

def configure_devices(model, device_ids):
    ''' Method to enforce parallelism during model training '''
    if  len(device_ids) > 1: #isinstance(device_ids, list): # for multiple GPU training
        print(f'Using multiple GPUS: {device_ids}')
        base_device = 'cuda:{}'.format(device_ids[0])
        model.to(base_device)
        model = nn.DataParallel(model, device_ids=device_ids)
    elif len(device_ids) == 1 and device_ids[0].isdigit(): # for single GPU training
        print(f'Using GPU ', device_ids[0])
        base_device = 'cuda:' + device_ids[0]
        model = model.to(base_device)
    else:
        print('Using CPU')
        base_device = 'cpu' # for CPU based training
        model = model.to('cpu')

    return base_device

# Loading in models

def load_vitmae_from_from_pretrained(model_path, pretraining, classification, classes):

    feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)

    if pretraining:
        model = ViTMAEForPreTraining.from_pretrained(model_path)
    elif classification:
        model = ViTForImageClassification.from_pretrained(model_path, num_labels=classes)

    return feature_extractor, model

def load_vitmae_from_arch_weights(model_arch, model_weights):

    '''
    As it stands, we can't load in a classifier with this method, so we won't worry about it.
    9/25
    '''

    # build model
    model = getattr(models_mae, model_arch)()
    print(model)
    # load model
    checkpoint = torch.load(model_weights, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    # print(msg)

    return model

def load_vitmae_from_from_pretrained_w_weights(from_pretrained_model_path, weights_path, pretraining, classification, classes):

    feature_extractor = ViTFeatureExtractor.from_pretrained(from_pretrained_model_path)

    if pretraining:
        model = ViTMAEForPreTraining.from_pretrained(from_pretrained_model_path)
        if weights_path != 'Scratch':
            checkpoint = torch.load(weights_path, map_location='cpu')
            msg = model.load_state_dict(checkpoint, strict=False)
            print(msg)
        else:
            config_file = model.config
            model_from_config = ViTMAEForPreTraining._from_config(config_file)
            model = model_from_config
    elif classification:
        model = ViTForImageClassification.from_pretrained(from_pretrained_model_path, num_labels=classes)
        if weights_path != 'Scratch':
            '''
            This could be cleaned up so we only load in the encoder and we don't have such a long error message.
            '''
            checkpoint = torch.load(weights_path, map_location='cpu')
            msg = model.load_state_dict(checkpoint, strict=False)
            print(msg)
        else:
            config_file = model.config
            model_from_config = ViTForImageClassification._from_config(config_file)
            model = model_from_config

    return feature_extractor, model

# Data preparation

def create_dataset_image_label(image_paths, label_paths):
        
    valid_image_paths = []
    valid_label_paths = []

    for i in range(len(image_paths)): # Sometimes the images paths don't open, so we need to weed those out
        if image_path.endswith('.npy'):
            try:
                img = np.load(image_path)
                valid_image_paths.append(image_paths[i])
                valid_label_paths.append(label_paths[i])
            except Exception as e:
                print(f"Error opening image '{image_paths[i]}': {e}")        
        else:    
            try:
                img = PImage.open(image_paths[i])
                img.verify()
                img.close()  # Close the image to release resources
                valid_image_paths.append(image_paths[i])
                valid_label_paths.append(label_paths[i])
            except Exception as e:
                print(f"Error opening image '{image_paths[i]}': {e}")

    dataset = Dataset.from_dict({"image": sorted(valid_image_paths),
                                "label": sorted(valid_label_paths)})
    dataset = dataset.cast_column("image", Image())
    # dataset = dataset.cast_column("label", torch.int32) # Do we need this? I don't think so.

    return dataset

def create_dataset_image_only(image_paths):
    valid_image_paths = []

    for image_path in image_paths:
        if image_path.endswith('.npy'):
            try:
                img = np.load(image_path)
                valid_image_paths.append(image_path)
            except Exception as e:
                print(f"Error opening image '{image_path}': {e}")        
        else:    
            try:
                img = PImage.open(image_path)
                img.verify()
                img.close()  # Close the image to release resources
                valid_image_paths.append(image_path)
            except Exception as e:
                print(f"Error opening image '{image_path}': {e}")

    dataset = Dataset.from_dict({"image": sorted(valid_image_paths)})
    dataset = dataset.cast_column("image", Image())

    return dataset

def get_train_val_splits(data_location, val_pct, num_rand_images):

    df = pd.read_csv(data_location)

    if num_rand_images is not None:
        df = df.sample(n=num_rand_images)
        df.reset_index(drop=True, inplace=True)

    # Sample a percentage of the DataFrame for validation
    val_df = df.sample(frac=val_pct, random_state=42)  # You can specify a random state for reproducibility

    # The rest of the data will be used for training
    train_df = df.drop(val_df.index)

    val_df.reset_index(drop=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    return train_df, val_df

class CustomImageDatasetFromCSV(Dataset):
    def __init__(self, data_frame, transform=None):
        self.data_frame = data_frame
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_path = self.data_frame.loc[idx]['image']
        
        if img_path.endswith('.npy'):
            # Load numpy array
            img = np.load(img_path)
            img = np.uint8(255*img)
            img = Image.fromarray(img, 'RGB')
        else:
            try:
                img = Image.open(img_path).convert('RGB')
            except OSError as e:
                print(f"Error loading image at path: {img_path}. OSError: {e}")
                img = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform is not None:
            img = self.transform(img)
        
        # Since this is for self-supervised learning, we don't need to provide target labels
        # We can just return None or any arbitrary value as the "target" or nothing at all
        return img

# Loss Over Dataset (Also in /sddata/projects/SSL/custom_mae/src/vitmae_loss_over_dataset.py)

def calculate_average_loss_over_dataset(model, base_device, train_dataloader):

    overall_loss = 0

    for batch in train_dataloader:

        input = batch['pixel_values']

        outputs = model(input.to(base_device))
        try:
            loss_on_batch = outputs['loss']
        except:
            loss_on_batch = outputs[0]

        overall_loss += loss_on_batch.detach().cpu()

    return overall_loss/len(train_dataloader)

# Visualization

def show_image(image, feature_extractor, title=''):
    # image is [H, W, 3]
    try: # If the feature_extractor is really a feature extractor
        feature_extractor_means = np.array(feature_extractor.image_mean)
        feature_extractor_stand_devs = np.array(feature_extractor.image_std)
        assert image.shape[2] == 3
        plt.imshow(torch.clip((image * feature_extractor_means + feature_extractor_stand_devs) * 255, 0, 255).int())
        plt.title(title, fontsize=16)
        plt.axis('off')
    except: # If not, then it is a tuple of (means, stand_devs)
        assert image.shape[2] == 3
        plt.imshow(torch.clip((image * feature_extractor[1] + feature_extractor[0]) * 255, 0, 255).int())
        plt.title(title, fontsize=16)
        plt.axis('off')

    return

def visualize(epoch, pixel_values, model, feature_extractor, show, save, dir_to_save, title):
    # forward pass
    outputs = model(pixel_values) # We note here that using ViTMAEForPreTraining we get a dictionary of odict_keys(['loss', 'logits', 'mask', 'ids_restore'])
    # and if using prepare model we get a tuple of (loss, y = logits, mask), so we need to take both into account
    try:
        y = model.unpatchify(outputs.logits)
    except:
        y = outputs[1]
        y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    
    # visualize the mask
    try:
        mask = outputs.mask.detach()
    except:
        mask = outputs[2]
    try:
        mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 *3)  # (N, H*W, p*p*3)
    except:
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', pixel_values).detach().cpu()

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], feature_extractor, "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], feature_extractor, "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], feature_extractor, "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], feature_extractor, "reconstruction + visible")

    if show:

        plt.show()

    if save:

        if epoch != None:

            plt.savefig(os.path.join(dir_to_save, title + '_' + str(epoch) + '_visualization.png'))

        else:

            plt.savefig(os.path.join(dir_to_save, title + '_visualization.png'))