#########
# Imports
#########

import numpy as np
import pandas as pd
import os
from shutil import copyfile
import sys

###########
# Functions
###########

def create_train_val_test_from_data(csv_path, current_image_location, image_col, split_col, path_to_save):

    image_split_df = pd.read_csv(csv_path)

    path_to_save_train = os.path.join(path_to_save, 'train')
    path_to_save_val = os.path.join(path_to_save, 'val')
    path_to_save_test = os.path.join(path_to_save, 'test')

    for path in [path_to_save_train, path_to_save_val, path_to_save_test]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory '{path}' created.")
        else:
            print(f"Directory '{path}' already exists.")

    for i in range(len(image_split_df)):
        image_path = image_split_df[image_col][i]
        split_label = image_split_df[split_col][i]

        image_path = os.path.join(current_image_location, os.path.basename(image_path))
        # print(image_path)
        # sys.exit()

        if split_label == 'train':
            save_path = os.path.join(path_to_save_train, os.path.basename(image_path))
        elif split_label == 'val':
            save_path = os.path.join(path_to_save_val, os.path.basename(image_path))
        elif split_label == 'test':
            save_path = os.path.join(path_to_save_test, os.path.basename(image_path))
        else:
            # Handle the case where the split label is not recognized
            print(f"Warning: Split label '{split_label}' not recognized. Skipping file: {image_path}")
            continue
        
        # Copy the image to the appropriate directory
        try:
            copyfile(image_path, save_path)
            # print(f"Image '{image_path}' saved to '{save_path}'.")
        except Exception as e:
            print(f"Error saving image '{image_path}' to '{save_path}': {e}")


if __name__ == '__main__':

    csv_path = '/sddata/projects/Cervical_Cancer_Projects/cervical_cancer/csvs/model_36_split_df_all_gt.csv'
    current_image_location = '/sddata/projects/Cervical_Cancer_Projects/data/full_dataset/full_dataset_duke_liger_itoju_5StLowQual'
    image_col = 'MASKED_IMG_ID'
    split_col = 'dataset'
    path_to_save = '/sddata/projects/Cervical_Cancer_Projects/data/full_dataset_duke_liger_train_val_test'

    create_train_val_test_from_data(csv_path, current_image_location, image_col, split_col, path_to_save)