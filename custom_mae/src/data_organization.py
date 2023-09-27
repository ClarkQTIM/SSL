#########
# Imports
#########

import numpy as np
import pandas as pd
import os
import glob



####################
# Running the Script
####################

if __name__ == '__main__':

    root_directory = '/data/retina_datasets'
    # Define the list of image file extensions you want to collect
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']

    # Initialize an empty list to store the image file paths
    image_paths = []

    # Recursively traverse the directory and collect image file paths
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(root_directory, '**', extension), recursive=True))

