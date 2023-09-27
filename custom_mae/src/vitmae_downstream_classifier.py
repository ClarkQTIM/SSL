#########
# Imports
#########

# Standard
import argparse
import json
import sys
sys.path.append('/sddata/projects/SSL/custom_mae/src')

# DL
import torch

# Py Files
from utils import *
from vitmae_downstream_classification import *

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
    parser.add_argument('-data_dir', '--data_dir', help='The directory where the images are', type=str, default = 'None') 
    parser.add_argument('-data_csv', '--data_csv', help='The prepared dataset in csv form for this training', type=str, default = 'None') 
    parser.add_argument('-image_col', '--image_col', help='The image column name in the csv, if applicable', type=str, default = 'None') 
    parser.add_argument('-label_col', '--label_col', help='The label column name in the csv, if applicable', type=str, default = 'None') 
    parser.add_argument('-val_pct', '--val_pct', help='The percentage to leave for validation', type=float, default = 0.1) 
    parser.add_argument('-batch', '--batch-size', help='Batch size for training.', type=int, default=16)
    parser.add_argument('-epochs', '--epochs', help='Number of epochs to train network.', type=int, default=100)
    parser.add_argument('-lr', '--learning-rate', help='Learning rate to use during training.', default=1e-4, type=float)
    parser.add_argument('-save_dir', '--model_directory', help='Directory where all files related to training are saved.', type=str, default='.')
    parser.add_argument('-title', '--title', help='Title of the plot.', type=str, default='Title')
    parser.add_argument('-gpus', '--gpus', help='GPU devices ids to use for parallel training', nargs='*', default=None)
    parser.add_argument('-seed', '--seed', help='Set seed to ensure reproducible training.', type=int, default=0) 

    args, extras = parser.parse_known_args()
    print('Args', args)

    # Model and dataset creation
    batch_size = args.batch_size

    from_pretrained_model = args.model_type
    model_checkpoint = None
    model_arch = args.model_arch

    if from_pretrained_model != None and model_checkpoint == None:
        print('We are loading in the model with from_pretrained.')

        feature_extractor, model = load_vitmae_from_from_pretrained(from_pretrained_model, False, True, 3)
        dataset = prepare_ds_from_csv_and_image_dir(args.data_dir, args.data_csv, args.image_col, args.label_col)

    elif from_pretrained_model != None and model_checkpoint != None:
        '''
        9/25: Not in use currently!
        '''
        print('We are loading in the model architecture from from_pretrained and also replacing the weights.')

        feature_extractor, model = load_vitmae_from_from_pretrained_w_weights(from_pretrained_model, model_checkpoint, False, True, 3)
        dataset = prepare_ds_from_csv_and_image_dir(args.data_dir, args.data_csv, args.image_col, args.label_col)

    else:
        print(f'There is an issue somewhere. The from_pretrained is {from_pretrained_model}, the architecture is {model_arch}, and the checkpoint is {model_checkpoint}.')


    def feat_extract_transform(example_batch):
        # Take a list of PIL images and turn them to pixel values
        inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt') # The same as above, but for everything in the batch

        # Don't forget to include the labels!
        inputs['label'] = example_batch['label']

        return inputs
    
    prepared_dataset = transform_dataset(dataset, feat_extract_transform)
    train_ds = prepared_dataset['train']
    val_ds = prepared_dataset['val']
    train_dataloader = DataLoader(train_ds, batch_size = batch_size, shuffle = False)
    val_dataloader = DataLoader(val_ds, batch_size = batch_size, shuffle = False)

    # Devices
    base_device = configure_devices(model, args.gpus)

    # Training args
    if not os.path.exists(args.model_directory):
        os.makedirs(args.model_directory)

    training_args = {
        "Data Dir": args.data_dir,
        "Data_csv": args.data_csv,
        "Image_Column": args.image_col,
        "Label_Column": args.label_col,
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
                    args.epochs, 
                    train_dataloader, 
                    val_dataloader,
                    args.learning_rate, 
                    args.model_directory, 
                    args.title,
                    args.seed)

    trainer.fit()