# Imports

import sys
# sys.path.append('/home/clachris/Documents/projects/DIET_SSL/')
# print(sys.path)

from models import __supported_networks__
from dataset_organization import *
from train_pretext import *
import argparse

# Full training pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training module')
    parser.add_argument('-m', '--model', help='Model type', type=str, choices=__supported_networks__.keys())
    parser.add_argument('-i', '--image-csv', help='Directory where the images are', type=str, default='csvs/dysis_cervix_images_paths.csv')
    # parser.add_argument('-h', '--scheduler', help='Learning Rate Scheduler type', type=str, choices=__supported_schedulers__.keys())
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
    ### Augs/Transformations
    batch_size = args.batch_size
    print(args.augs)
    transforms, augs_list = full_transformations(args.augs)
    print('aug list', augs_list)

    ### Image paths, label creation, and dataloading
    if 'cifar100' in args.image_csv:
        X = load_cifar_paths_diet_ssl(args.image_csv)
    elif 'dysis' in args.image_csv:
        X = load_dysis_paths_diet_ssl(args.image_csv)
    elif 'iris' in args.image_csv:
        X = load_iris_paths_diet_ssl(args.image_csv)
        
    Y = list(range(0, len(X)))
    print(f'There are {len(X)} images.')
    train_dataloader = Dataset_Dataloader(X, Y, transforms, batch_size)

    ## Model
    model_type = args.model
    epochs = args.epochs
    learning_rate = args.learning_rate

    model = __supported_networks__[model_type]
    K = model.fc.in_features
    model.fc= nn.Identity()

    W = nn.Linear(K, len(Y), bias = False)

    ## Saving the training parameters
    training_args = {
        "Model_type": model_type,
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
                    W,
                    epochs, 
                    train_dataloader, 
                    learning_rate, 
                    args.save_dir, 
                    args.save_title,
                    args.plotting,
                    args.results_saving
    )
    
    trainer.configure_devices(args.gpus)

    ## Training
    trainer.fit()