#########
# Imports
#########

# Standard
import os
import numpy as np
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed
from datetime import datetime
import sys

# Image Manipulation
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True # Issues with some images 'OSError: image file is truncated (n bytes not processed)'

###########
# Functions
###########

def find_file_path(root_directory, target_file_name):
    for root, dirs, files in os.walk(root_directory):
        if target_file_name in files:
            return os.path.join(root, target_file_name)

    return None

def calculate_mean_std_parallel(img_path):
    try:
        # img = Image.open(img_path)
        img = np.load(img_path)
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        shape = img_array.shape
        sum_channels = np.sum(img_array, axis=(0, 1))
        sum_squared_channels = np.sum(img_array ** 2, axis=(0, 1))
        return sum_channels, sum_squared_channels, shape
    except Exception as e:
        print(f"Error opening/verifying image '{img_path}': {e}")
        return None

def calculate_mean_std(image_paths):
    num_images = len(image_paths)
    sum_channels = np.zeros((3,), dtype=np.float64)
    sum_squared_channels = np.zeros((3,), dtype=np.float64)
    shapes = []

    # Results
    results = Parallel(n_jobs=-1)(delayed(calculate_mean_std_parallel)(img_path) for img_path in image_paths)
    valid_results = [res for res in results if res is not None]

    for sum_chan, sum_sq_chan, shape in valid_results:
        sum_channels += sum_chan
        sum_squared_channels += sum_sq_chan
        shapes.append(shape)

    # Consider individual image sizes for the mean and std calculation
    total_pixels = sum(shape[0] * shape[1] for shape in shapes)
    
    mean_channels = sum_channels / total_pixels
    std_channels = np.sqrt(
        (sum_squared_channels / total_pixels) - (mean_channels ** 2)
    )

    return mean_channels, std_channels, len(image_paths)

def normalization_stats_from_dir(images_origin, dir_to_find_images, image_col, split_col, train_label, val_label, save_path):

    # Data
    current_date = datetime.now()
    date_string = current_date.strftime('%Y-%m-%d %H:%M:%S')

    # Paths
    if images_origin.split('.')[-1] != 'csv':
        images = os.listdir(images_origin)
        print(len(images))
        exclude_df = pd.read_csv('/sddata/projects/SSL/custom_mae/csvs/model_36_split_df_test1_only.csv')
        print(len(list(exclude_df[image_col])))
        image_paths = [image for image in images if image not in list(exclude_df[image_col])]
        image_paths = [os.path.join(images_origin, image) for image in image_paths]
        print(len(image_paths))
    else:
        print('Reading from csv')
        data_df = pd.read_csv(images_origin)
        train_data_df = data_df[(data_df[split_col] == train_label)][image_col]
        val_data_df = data_df[(data_df[split_col] == val_label)]
        image_paths = train_data_df
        print(image_paths)
        # image_paths = list(train_data_df[image_col])

        # for i in range(len(train_data_df)):
        #     image_path = list(train_data_df[image_col])[i]
        #     current_train_dir = dir_to_find_images
        #     image_path = os.path.join(current_train_dir, image_path.split('/')[-1])
        #     image_paths.append(image_path)

        # for i in range(len(val_data_df)):
        #     image_path = list(val_data_df[image_col])[i]
        #     current_val_dir = dir_to_find_images
        #     image_path = os.path.join(current_val_dir, image_path.split('/')[-1])
        #     image_paths.append(image_path)

    print(len(image_paths))
    # Means and Standard Deviations
    mean_channels, std_channels, num_images = calculate_mean_std(image_paths)

    # Dataframe
    normalize_dict = {'Data Origin': images_origin, 'Date': date_string, 'Num_Images': num_images, 'Means': np.round(mean_channels, 3), 'Standard_Deviations': np.round(std_channels, 3)}
    normalize_df = pd.DataFrame(normalize_dict)
    normalize_df.to_csv(save_path)

if __name__ == "__main__":

    data_dir = '/sddata/projects/SSL/csvs/datasets/dmist_train_only.csv'
    dir_to_find_images = '/sddata/projects/Cervical_Cancer_Projects/data/SEED/'
    save_path = '/sddata/projects/SSL/csvs/norms/dmist_train_only.csv'
    # save_path = 'None'

    normalization_stats_from_dir(data_dir, dir_to_find_images, 'Image', 'dataset', 'train', 'val', save_path)