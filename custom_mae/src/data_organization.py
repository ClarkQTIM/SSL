#########
# Imports
#########

import numpy as np
import pandas as pd
import os
import glob

###########
# Functions
###########

def create_dia_ret_csv(data_dir, save_path):

    training_data = os.listdir(os.path.join(data_dir, 'training'))
    training_data.sort()
    training_data = [os.path.join(data_dir, 'train/' + image) for image in training_data]

    testing_data = os.listdir(os.path.join(data_dir, 'test'))
    testing_data.sort()
    testing_data = [os.path.join(data_dir, 'test/' + image) for image in testing_data]

    data_dict = {'Image': training_data + testing_data}
    data_df = pd.DataFrame(data_dict)
    data_df.to_csv(save_path)


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

    data_dir = ''
    save_path = '/sddata/projects/SSL/custom_mae/csvs/all_dr_images.csv'
    create_dia_ret_csv(data_dir, save_path)

