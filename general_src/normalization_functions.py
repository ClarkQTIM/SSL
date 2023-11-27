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

# Image Manipulation
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True # Issues with some images 'OSError: image file is truncated (n bytes not processed)'

###########
# Functions
###########

def calculate_mean_std_parallel(img_path):
    try:
        img = Image.open(img_path)
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

    return mean_channels, std_channels

def normalization_stats_from_dir(images_origin, image_col, save_path):

    # Data
    current_date = datetime.now()
    date_string = current_date.strftime('%Y-%m-%d %H:%M:%S')

    # Paths
    if images_origin.split('.')[-1] != 'csv':
        images = os.listdir(images_origin)
        image_paths = [os.path.join(images_origin, image) for image in images]
    else:
        data_df = pd.read_csv(images_origin)
        data_df = data_df[data_df['dataset'] == 'train']
        image_paths = data_df[image_col].tolist()
        image_paths = [os.path.join('/sddata/projects/Cervical_Cancer_Projects/data/full_dataset_duke_liger_itoju_5StLowQual', image) for image in image_paths]

    # Means and Standard Deviations
    mean_channels, std_channels = calculate_mean_std(image_paths)

    # Dataframe
    normalize_dict = {'Data Origin': images_origin, 'Date': date_string, 'Means': np.round(mean_channels,3), 'Standard_Deviations': np.round(std_channels,3)}
    normalize_df = pd.DataFrame(normalize_dict)
    normalize_df.to_csv(save_path)

if __name__ == "__main__":

    data_dir = '/sddata/projects/Cervical_Cancer_Projects/cervical_cancer/csvs/full_dataset_duke_liger_split_df.csv'
    save_path = '/sddata/projects/SSL/csvs/full_dataset_duke_liger_itoju_5StLowQual_split_df_train_norms.csv'

    normalization_stats_from_dir(data_dir, 'MASKED_IMG_ID', save_path)