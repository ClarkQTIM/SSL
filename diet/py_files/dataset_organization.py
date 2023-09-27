# Imports

## Standard
import os
import pandas as pd
import numpy as np
from PIL import Image
import json

## MONAI and Torch
import torch
from monai.data import DataLoader
from monai.utils.misc import first

## Image Manipulation
from skimage.transform import resize
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    Resize,
    LoadImage,
    RandFlip,
    RandRotate,
    RandGaussianSmooth,
    ScaleIntensity,
    ToTensor,
)
from torchvision.transforms import (
    RandomResizedCrop,
    ColorJitter,
    RandomApply,
)

## Label Encoding
from sklearn import preprocessing

# Creating a csv file with the patients as rows and time series as columns

def create_dysis_csv(path_to_dataset, path_to_save):

    patients = os.listdir(path_to_dataset) # Getting the patients
    patients.sort() # Sorting them, as they might be out of order
    df = pd.DataFrame(columns=range(0,17)) # There are 17 images for each patient
    for idx in range(len(patients)): # Adding the full path to each image in the sequence as a row in the dataframe
        path_to_patient = os.path.join(path_to_dataset, patients[idx])
        images = os.listdir(path_to_patient)
        images.sort()
        image_paths = [os.path.join(path_to_patient, img) for img in images]
        df.loc[idx] = image_paths
    df.to_csv(path_to_save)

def create_cifar_diet_ssl_csv(path_to_dataset, path_to_save):

    X = list()
    Y = list()
    Z = list()

    for dataset_type in ['train/', 'test/']:

        class_archetypes = os.listdir(path_to_dataset + dataset_type)
        class_archetypes.sort()

        for class_archetype in class_archetypes: # Animal type
            classes = os.listdir(path_to_dataset + dataset_type + class_archetype)
            classes.sort()
            for class_type in classes: # Animal subtype
                paths = os.listdir(path_to_dataset + dataset_type + class_archetype + '/' + class_type)
                paths.sort()
                for path in paths:
                    full_path = path_to_dataset + dataset_type + class_archetype + '/' + class_type + '/' + path
                    X.append(full_path)
                    Y.append(class_type)
                    Z.append(dataset_type[:-1]) # The [-1] is to remove the '/' at the end
    cifar100_dict = {'img_path': X, 'class': Y, 'division_type': Z}
    cifar100_df = pd.DataFrame(cifar100_dict)
    cifar100_df.to_csv(path_to_save)

def create_iris_diet_ssl_csv(path_to_dataset, path_to_save):
    '''
    Finish this up
    '''

if __name__ == '__main__': # Run this script to generate this csv
    create_dysis_csv('/data/cervix_datasets/JayashreeSample/lesion_set_1/', 'csvs/dysis_cervix_images_paths.csv')
    create_cifar_diet_ssl_csv('/data/other_datasets/cifar100/', 'csvs/cifar100_paths_labels.csv')
    '''
    Add iris
    '''

# Load paths and classes from csv

def load_dysis_paths_diet_ssl(dysis_csv):

    # Loading and organizing the dataframe
    dysis_df = pd.read_csv(dysis_csv)
    dysis_df = dysis_df.drop('Unnamed: 0', axis = 1)

    # Since we don't care about the multiple timepoints, so we stack all the columns after having removed the 'Unnamed: 0'
    dysis_df = dysis_df.stack(-1)
    img_paths = list(dysis_df)

    return img_paths

def load_cifar_paths_diet_ssl(cifar100_csv):

    cifar_df = pd.read_csv(cifar100_csv)
    img_paths = list(cifar_df['img_path']) # We only care about the image paths so we can ignore the other columns

    return img_paths

def load_cifar_paths_classes(cifar100_csv):

    cifar_df = pd.read_csv(cifar100_csv)
    cifar_df = cifar_df.drop('Unnamed: 0', axis = 1)

    # Train X and Y
    cifar_df_train = cifar_df.loc[cifar_df['division_type'] == 'train']
    img_train_paths = list(cifar_df_train['img_path'])
    classes_train = list(cifar_df_train['class'])
    # Test X and Y
    cifar_df_test = cifar_df.loc[cifar_df['division_type'] == 'test']
    img_test_paths = list(cifar_df_test['img_path'])
    classes_test = list(cifar_df_test['class'])

    # Label Encoding
    le = preprocessing.LabelEncoder()

    le.fit(y = classes_train)
    train_Y = le.transform(classes_train)
    test_Y = le.transform(classes_test)

    return img_train_paths, train_Y, img_test_paths, test_Y

def load_iris_paths_diet_ssl(iris_csv):
    '''
    Finish this
    '''

# Organizing the data

def DataDivision(all_X, all_Y, train_pct, val_test_pct):

    try:

        train_X, val_X = torch.utils.data.random_split(all_X, [int(round(train_pct*len(all_X))), int(round(val_test_pct*len(all_X)))], generator=torch.Generator().manual_seed(0))
        train_Y, val_Y = torch.utils.data.random_split(all_Y, [int(round(train_pct*len(all_Y))), int(round(val_test_pct*len(all_Y)))], generator=torch.Generator().manual_seed(0))

        test_X, val_X = torch.utils.data.random_split(val_X, [int(0.5*len(val_X)), int(0.5*len(val_X))], generator=torch.Generator().manual_seed(0))
        test_Y, val_Y = torch.utils.data.random_split(val_Y, [int(0.5*len(val_Y)), int(0.5*len(val_Y))], generator=torch.Generator().manual_seed(0))

    except:

        train_X, val_X = torch.utils.data.random_split(all_X, [int(np.ceil(train_pct*len(all_X))), int(np.floor(val_test_pct*len(all_X)))], generator=torch.Generator().manual_seed(0))
        train_Y, val_Y = torch.utils.data.random_split(all_Y, [int(np.ceil(train_pct*len(all_Y))), int(np.floor(val_test_pct*len(all_Y)))], generator=torch.Generator().manual_seed(0))

        test_X, val_X = torch.utils.data.random_split(val_X, [int(np.ceil(0.5*len(val_X))), int(np.floor(0.5*len(val_X)))], generator=torch.Generator().manual_seed(0))
        test_Y, val_Y = torch.utils.data.random_split(val_Y, [int(np.ceil(0.5*len(val_Y))), int(np.floor(0.5*len(val_Y)))], generator=torch.Generator().manual_seed(0))

    return train_X, train_Y, val_X, val_Y, test_X, test_Y

# Augmentations

## Possible Augmentations
__augs__ = { 
    # Resize
    'resize': Resize([512, 512]),
    # Spatial
    'random_flip_x': RandFlip(spatial_axis = 1, prob = 1), 
    'random_flip_y': RandFlip(spatial_axis = 0, prob = 1),
    'random_resize_crop': RandomResizedCrop(size=512, scale=(0.80,1)),
    # Intensity
    'gaussian_blur': RandGaussianSmooth(prob=0.5),
    'color_jitter': RandomApply([ColorJitter(brightness=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.05, 0.05), contrast=(0.5, 1.5))], p=0.5),
    'random_rotation': RandRotate(range_x=15, prob=0.5, keep_size=True, padding_mode='zeros')
    }

# Getting our augmentations/transformations from the json file. This will feed into the following function.
def get_augmentations_from_json(augs):

    # Getting a list of our augmentations
    augs_list = []
    desc_list = []
    for aug in augs.keys():
        if augs[aug]:
            print('Adding data augmentation: ', aug)
            augs_list.append(__augs__[aug])
            desc_list.append(aug)

    # Prepending the loading and ensuring channel first transformations
    augs_list.insert(0, LoadImage(image_only = True)) # We are giving the dataloader only the image paths, so we need to open the images first              
    augs_list.insert(1, EnsureChannelFirst()) # This will make it so the channel is first (in this case, this will be (3, 512, 512)) OR will add one to a 2D image to get (1, 512, 512)
    # Appending the scaling and totensor transformations
    augs_list.append(ScaleIntensity()) # Scaling the pixel values to [0,1]
    augs_list.append(ToTensor()) # Making sure it is a tensor

    return augs_list, desc_list # Returning the keys for training tracking

# Getting our full set of transformations, i.e., adding the augmentations to normal reading/scaling, etc. transformations
def full_transformations(augs):
    if isinstance(augs, str) and augs.endswith('.json'):
        augs_dict = json.load(open(augs))
        augs_list, key_list  = get_augmentations_from_json(augs_dict)

    elif isinstance(augs, dict):
        augs_dict = augs
        augs_list, key_list = get_augmentations_from_json(augs)

    else:
        augs_list = [LoadImage(image_only = True), # We are giving the dataloader only the image paths, so we need to open the images first
                EnsureChannelFirst(), # This will make it so the channel is first (in this case, this will be (3, 512, 512)) OR will add one to a 2D image to get (1, 512, 512)
                ScaleIntensity(), # Scaling the pixel values to [0,1]
                ToTensor() # Making sure it is a tensor
                ]
        key_list = []
    return Compose(augs_list), key_list # Returning 

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]
    
def Dataset_Dataloader(X, Y, transforms, batch_size):
    dataset = ClassificationDataset(X, Y, transforms)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    
    return dataloader

def conf(dataloader, printing):
    conf = first(dataloader)
    if printing:
        print('The number of batches and the shapes of the first and second items in the dataloader batch are:', len(dataloader), conf[0].numpy().shape, conf[1].numpy().shape) 

    return conf