#########
# Imports
#########

# Standard
import argparse
import json
import sys

# DL
import torch

# Py Files
import models_mae
from utils import *
from vitmae_finetune import *


########################
# Full training pipeline
########################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training module')
    parser.add_argument('-model_type', '--model-type', help='Name of the model network to use.', type=str, default='facebook/vit-mae-base')
    parser.add_argument('-model_arch', '--model-arch', help='Architecture of the model network to use.', type=str, default='mae_vit_large_patch16')
    parser.add_argument('-model_checkpt', '--model-checkpt', help='Model checkpoint to load', type=str, default='None')
    parser.add_argument('-means', '--means', help='Mean values if not from feature_extractor', type=str, default='None')
    parser.add_argument('-stand_devs', '--stand_devs', help='Standard deviation values if not from feature_extractor', type=str, default='None')
    parser.add_argument('-size', '--size', help='Resizing value if not from feature_extractor', type=int, default='None')
    parser.add_argument('-data', '--dataset', help='The prepared dataset (in csv form or as a directory) for this training', type=str, default = 'None') 
    parser.add_argument('-col', '--col', help='The image column name in the csv, if applicable', type=str, default = 'None') 
    parser.add_argument('-val_pct', '--val_pct', help='The percentage to leave for validation', type=float, default = 0.1) 
    parser.add_argument('-batch', '--batch-size', help='Batch size for training.', type=int, default=16)
    parser.add_argument('-epochs', '--epochs', help='Number of epochs to train network.', type=int, default=100)
    parser.add_argument('-lr', '--learning-rate', help='Learning rate to use during training.', default=1e-4, type=float)
    parser.add_argument('-vis', '--visualization', help='Whether to plot and save the attention weights.', type=int, default=0)
    parser.add_argument('-save_dir', '--model_directory', help='Directory where all files related to training are saved.', type=str, default='.')
    parser.add_argument('-title', '--title', help='Title of the plot.', type=str, default='Title')
    parser.add_argument('-gpus', '--gpus', help='GPU devices ids to use for parallel training', nargs='*', default=None)
    parser.add_argument('-seed', '--seed', help='Set seed to ensure reproducible training.', type=int, default=0) 

    args, extras = parser.parse_known_args()
    print('Args', args)

    # Model and dataset creation
    data_dir = args.dataset
    batch_size = args.batch_size
    col = args.col

    try:
        model = load_vitmae_from_arch_weights(args.model_arch, args.model_checkpt)
        print('We are loading a model from an architecture and model weights.')
        means = np.array([float(x) for x in args.means.split('_')])
        stand_devs = np.array([float(x) for x in args.stand_devs.split('_')])
        size = args.size
        prepared_dataset, valid_image_paths = prepare_dataset_from_dir_parallel(data_dir, col, None, size, means, stand_devs, None)
        feature_extractor = (means, stand_devs)

    except:
        feature_extractor, model = load_vitmae_from_from_pretrained(args.model_type, True, False, None)
        print('We are loading a model from from_pretrained.')
        prepared_dataset, valid_image_paths = prepare_dataset_from_dir_parallel(data_dir, col, feature_extractor, None, None, None, None)

    train_ds, val_ds = data_division(prepared_dataset, args.val_pct, args.seed)
    train_dataloader = DataLoader([prepped_data.squeeze(0) for prepped_data in train_ds], batch_size = batch_size, shuffle = False)
    val_dataloader = DataLoader([prepped_data.squeeze(0) for prepped_data in val_ds], batch_size = batch_size, shuffle = False)
    
    # Devices
    base_device = configure_devices(model, args.gpus)

    # Training args
    if not os.path.exists(args.model_directory):
        os.makedirs(args.model_directory)

    training_args = {
        "Dataset": args.dataset,
        "Image_Column": args.col,
        "Model_Type": args.model_type,
        "Model_Arch": args.model_arch,
        "Checkpoint": args.model_checkpt,
        "Batch_Size": args.batch_size,
        "Epochs": args.epochs,
        "Learning_Rate": args.learning_rate,
        "Scheduler": 'None',
        "Val_Pct": args.val_pct,
        "Seed": args.seed
    }
    savepath = os.path.join(args.model_directory, args.title + "_training_args.json")
    with open(savepath, "w") as f:
        json.dump(training_args, f)

    # # Saving the csv used to train
    # dataset = pd.read_csv(args.dataset)
    # dataset.to_csv(args.model_directory + args.dataset.split('/')[-1])

    # Training
    trainer = Train(model,
                    base_device,
                    feature_extractor,
                    args.epochs, 
                    train_dataloader, 
                    val_dataloader,
                    args.learning_rate, 
                    None,
                    args.visualization,
                    args.model_directory, 
                    args.title,
                    args.seed)

    trainer.fit()