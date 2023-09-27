# Imports

import sys
# sys.path.append('/home/clachris/Documents/projects/DIET_SSL/')
# print(sys.path)

from models import __supported_networks__
from dataset_organization import *
from train_downstream import *
import argparse

# Full training pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training module')
    parser.add_argument('-m', '--model', help='Model type', type=str, choices=__supported_networks__.keys())
    parser.add_argument('-k', '--checkpoint', help='Model checkpoint to load', type=str, default='.')
    parser.add_argument('-i', '--image-csv', help='Directory where the images are', type=str, default='csvs/dysis_cervix_images_paths.csv')
    parser.add_argument('-b', '--batch-size', help='Batch size for training.', type=int, default=16)
    parser.add_argument('-e', '--epochs', help='Number of epochs to train network.', type=int, default=100)
    parser.add_argument('-r', '--learning-rate', help='Learning rate to use during training.', type=float, default=1e-4)
    parser.add_argument('-a', '--augs', help='Augmentations to use', type=str, default='None')
    parser.add_argument('-d', '--save_dir', help='Directory where all files related to training are saved.', type=str, default='.')
    parser.add_argument('-s', '--save_title', help='Title to add to the save files to distinguish them.', type=str, default='.')
    parser.add_argument('-p', '--plotting', help='Whether to save the plot of the results.', action='store_true')
    parser.add_argument('-v', '--results_saving', help='Whether to save the results (weights and epoch/val results).', action='store_true')
    parser.add_argument('-g', '--gpus', help='GPU devices ids to use for parallel training', nargs='*', default=None)
    parser.add_argument('--seed', help='Set seed to ensure reproducible training.', type=int, default=1234) 
    parser.add_argument('-c', '--cutoff', help='Cutoff for dataset', type=int, default=500)

    args, extras = parser.parse_known_args()

    ## Data Organization and Loading
    batch_size = args.batch_size
    print(args.augs)
    augmentations, augs_list = full_transformations(args.augs)
    print('aug list', augs_list)

    if 'cifar100' in args.image_csv:
        train_X, train_Y, test_X, test_Y = load_cifar_paths_classes(args.image_csv)
        train_dataloader = Dataset_Dataloader(train_X, train_Y, augmentations, batch_size)
        test_dataloader = Dataset_Dataloader(test_X, test_Y, augmentations, batch_size)
    else:
        print('For now, this is only set up for Cifar100, so please put in the the correct csv for that.')

    ## Model
    model_type = args.model
    epochs = args.epochs
    learning_rate = args.learning_rate

    ### Loading in the model
    model_checkpoint = args.checkpoint
    if model_checkpoint == 'None': # In this case, we are just loading in a blank model for testing or regular training
        model = __supported_networks__[model_type] # Loading in blank network
    else: # In this case, we are loading in the model architecture, changing out the last layer, loading the checkpoint, and re-setting the 
    # final layer to the original model's for fine-tuning on that layer/other weights
        model = __supported_networks__[model_type] # Loading in blank network
        fully_connected = model.fc # Grabbing the fully connected layer
        model.fc = nn.Identity() # Changing the last layer to match what we saved
        model.load_state_dict(torch.load(model_checkpoint)) # Loading in the weights
        model.fc = fully_connected # Replacing the nn.Identity() layer with the correct fully connected one

    ## Saving the training parameters
    training_args = {
        "Model_type": model_type,
        "Model Checkpoint": model_checkpoint,
        "Augmentations": augs_list,
        "Epochs": epochs,
        "Learning Rate": learning_rate,
        "Scheduler": 'cosineannealing',
        "Batch_Size": batch_size,
    }
    savepath = os.path.join(args.save_dir, args.save_title + "training_args.json")
    with open(savepath, "w") as f:
        json.dump(training_args, f)
    
    ## Trainer
    trainer = Train(model,
                    epochs, 
                    train_dataloader, 
                    test_dataloader,
                    learning_rate, 
                    args.save_dir, 
                    args.save_title,
                    args.plotting,
                    args.results_saving
    )
    
    trainer.configure_devices(args.gpus)

    ## Training
    trainer.fit()