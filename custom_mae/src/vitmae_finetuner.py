#########
# Imports
#########

# Standard
import argparse
import json
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # Issues with some images 'OSError: image file is truncated (n bytes not processed)'

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

    from_pretrained_model = args.model_type
    model_checkpoint = args.model_checkpt
    model_arch = args.model_arch

    num_rand_images = None

    if from_pretrained_model != 'None' and model_checkpoint == 'None':
        print('We are loading in the model with from_pretrained.')
        feature_extractor, model = load_vitmae_from_from_pretrained(from_pretrained_model, True, False, None)
        dataset = prepare_dataset_reconstruction(data_dir, col, args.val_pct, num_rand_images)

    elif from_pretrained_model != 'None' and model_checkpoint == 'Scratch':
        print('We are loading in an INITIALIZED model with the correct architecture. This is NOT fine-tuned.')
        feature_extractor, model = load_vitmae_from_from_pretrained_w_weights(from_pretrained_model, model_checkpoint, True, False, None)
        dataset = prepare_dataset_reconstruction(data_dir, col, args.val_pct, num_rand_images)

    elif from_pretrained_model != 'None' and model_checkpoint != 'None':
        print('We are loading in the model with from_pretrained and the swapping out the weights.')
        feature_extractor, model = load_vitmae_from_from_pretrained_w_weights(from_pretrained_model, model_checkpoint, True, False, None)
        dataset = prepare_dataset_reconstruction(data_dir, col, args.val_pct, num_rand_images)

    else:
        print('Issue loading in model for fine-tuning. Check arguments')
        sys.exit()

    def feat_extract_transform(example_batch):
        # Take a list of PIL images and turn them to pixel values
        inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt') # The same as above, but for everything in the batch

        return inputs

    prepared_dataset = transform_dataset(dataset, feat_extract_transform)
    train_ds = prepared_dataset['train']
    val_ds = prepared_dataset['val']
    print(f'We have {len(train_ds)} training examples and {len(val_ds)} validation ones.')
    train_dataloader = DataLoader(train_ds, batch_size = batch_size, shuffle = False)
    val_dataloader = DataLoader(val_ds, batch_size = batch_size, shuffle = False)

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
        "Scheduler": 'LinearLR', # Hardcoded for now
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