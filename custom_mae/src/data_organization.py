#########
# Imports
#########

import numpy as np
import pandas as pd
import os
import glob
import random

###########
# Functions
###########

def create_dia_ret_csv(data_dir, save_path1, save_path2):

    training_data = os.listdir(os.path.join(data_dir, 'training'))
    training_data.sort()
    training_data = [os.path.join(data_dir, 'training/' + image) for image in training_data]

    # Dividing the training data into train, test, val
    total_len = len(training_data)
    train_len = int(total_len * 0.8)
    val_len = int(total_len * 0.1)
    test_len = total_len - train_len - val_len

    # Create a list with corresponding labels
    data_split = ["train"] * train_len + ["val"] * val_len + ["test"] * test_len

    # Shuffle the list to randomize the order
    random.shuffle(data_split)

    testing_data = os.listdir(os.path.join(data_dir, 'test'))
    testing_data.sort()
    testing_data = [os.path.join(data_dir, 'test/' + image) for image in testing_data]
    testing_data_labels = ['true_test']*len(testing_data)

    data_dict = {'Image': training_data + testing_data, "Label": data_split + testing_data_labels}
    data_df = pd.DataFrame(data_dict)
    data_df.to_csv(save_path1)

    data_dict = {'Image': training_data, "Label": data_split}
    data_df = pd.DataFrame(data_dict)
    data_df.to_csv(save_path2)

def create_dr_train_val_csv(train_dir, val_dir, save_path):

    train_images = os.listdir(train_dir)
    train_images = [os.path.join(train_dir, image) for image in train_images]
    train_split = ['train'] * len(train_images)

    val_images = os.listdir(val_dir)
    val_images = [os.path.join(val_dir, image) for image in val_images]
    val_split = ['val'] * len(val_images)

    splits = train_split + val_split
    all_images = train_images + val_images

    csv_dict = {'Image':all_images, 'Split': splits}
    csv_df = pd.DataFrame(csv_dict)
    csv_df.to_csv(save_path)

####################
# Running the Script
####################

if __name__ == '__main__':

    # root_directory = '/data/retina_datasets'
    # # Define the list of image file extensions you want to collect
    # image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']

    # # Initialize an empty list to store the image file paths
    # image_paths = []

    # # Recursively traverse the directory and collect image file paths
    # for extension in image_extensions:
    #     image_paths.extend(glob.glob(os.path.join(root_directory, '**', extension), recursive=True))

    # data_dir = '/sddata/projects/SSL/data/diabetic_retinopathy_detection/'
    # save_path1 = '/sddata/projects/SSL/custom_mae/csvs/all_dr_images.csv'
    # save_path2 = '/sddata/projects/SSL/custom_mae/csvs/dr_training_images.csv'
    # create_dia_ret_csv(data_dir, save_path1, save_path2)

    train_dir = '/sddata/data/retina_datasets/diabetic_retinopathy_detection/data_with_labels_train_val/train'
    val_dir = '/sddata/data/retina_datasets/diabetic_retinopathy_detection/data_with_labels_train_val/val'
    save_path = '/sddata/projects/SSL/custom_mae/csvs/dia_ret_train_val_only.csv'
    create_dr_train_val_csv(train_dir, val_dir, save_path)