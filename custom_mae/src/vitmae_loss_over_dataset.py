#########
# Imports
#########

# Standard
import os
import json
import time
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # Issues with some images 'OSError: image file is truncated (n bytes not processed)'


# DL
from monai.data import DataLoader

# Py Files
from utils import *

###########
# Functions
###########

def calculate_average_loss_over_dataset(model, base_device, train_dataloader):

    overall_loss = 0

    for batch in train_dataloader:

        input = batch['pixel_values']

        outputs = model(input.to(base_device))
        try:
            loss_on_batch = outputs['loss']
        except:
            loss_on_batch = outputs[0]

        overall_loss += loss_on_batch.detach().cpu()

    return overall_loss/len(train_dataloader)

def save_config(dataset_path, from_pretrained_model, model_checkpoint, model_arch,
                means, stand_devs, size, batch_size, base_device, average_loss, dir_to_save, title):

    args_to_save = {
        "Dataset": dataset_path,
        "from_pretrained_model": from_pretrained_model,
        "model_checkpoint": model_checkpoint,
        "model_arch": model_arch,
        "Means": str(means),
        "Std_Devs": str(stand_devs),
        "Size": size,
        "Batch_Size": batch_size,
        "Base_Device": base_device,
        "Average_Loss": average_loss.item()
    }

    savepath = os.path.join(dir_to_save, title + "_loss_args.json")
    with open(savepath, "w") as f:
        json.dump(args_to_save, f)

###################
# Loss Over Dataset
###################

if __name__ == "__main__":

    data_dir = '/sddata/projects/Cervical_Cancer_Projects/data/full_dataset_duke_liger_itoju_5StLowQual'
    # data_dir = '/sddata/projects/SSL/custom_mae/csvs/all_cropped_fundus_images.csv'

    col = None

    from_pretrained_model = "facebook/vit-mae-huge"
    model_checkpoint = '/sddata/projects/SSL/custom_mae/Reconstruction_Custom_Finetuning_Best_Models/Facebook_Huge_Cervix_Custom_Finetune_v2_Cont_best_epoch.pth'
    # model_checkpoint =  '/sddata/projects/SSL/custom_mae/Reconstruction_Custom_Finetuning_Best_Models/Facebook_Huge_Cropped_Fundus_Custom_Finetune_best_epoch.pth'
    # model_arch = 'mae_vit_large_patch16'
    # model_checkpoint = None
    model_arch = None
    batch_size = 32
    title = 'Facebook_Huge_Finetuned_v2_Cont_full_dataset_duke_liger_itoju_5StLowQual_Average_Loss_loss'
    # title = 'testing'
    dir_to_save = '/sddata/projects/SSL/custom_mae/Reconstruction_Dataset_Losses'
    gpus = ['0']

    '''
    The following are from ImageNet
    '''

    # means = np.array([0.485, 0.456, 0.406])
    # stand_devs = np.array([0.229, 0.224, 0.225])
    # size = 224

    means = None
    stand_devs = None
    size = None
    num_rand_images = None # Using all our images

    start = time.time()

    print(data_dir, from_pretrained_model, model_checkpoint, model_arch,
          means, stand_devs, size, batch_size, dir_to_save, title)

    # Loading in our model and feature extractor
    if from_pretrained_model != None and model_checkpoint == None:
        print('We are loading in the model with from_pretrained.')

        feature_extractor, model = load_vitmae_from_from_pretrained(from_pretrained_model, True, False, None)
        base_device = configure_devices(model, gpus)
        # prepared_dataset, valid_image_paths = prepare_dataset_from_dir_parallel(data_dir, 'Image', feature_extractor, None, None, None, None)

    # elif model_arch != None and model_checkpoint != None:
    #     print('We are loading in the model from an architecture and weights.')

    #     model = load_vitmae_from_arch_weights(model_arch, model_checkpoint)
    #     base_device = configure_devices(model, gpus)
    #     prepared_dataset, valid_image_paths = prepare_dataset_from_dir_parallel(data_dir, 'Image', None, size, means, stand_devs, None)
    # Not currently in use. To use, we would need to then calculate the average loss from this line

    elif from_pretrained_model != None and model_checkpoint != None:
        print('We are loading in the model architecture from from_pretrained and also replacing the weights.')

        feature_extractor, model = load_vitmae_from_from_pretrained_w_weights(from_pretrained_model, model_checkpoint, True, False, None)
        base_device = configure_devices(model, gpus)

    else:
        print(f'There is an issue somewhere. The from_pretrained is {from_pretrained_model}, the architecture is {model_arch}, and the checkpoint is {model_checkpoint}.')

    # Creating our feature extractor function
    def feat_extract_transform(example_batch):
        # Take a list of PIL images and turn them to pixel values
        inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt') # The same as above, but for everything in the batch

        return inputs

    # Training dataset creation
    dataset = prepare_dataset_reconstruction(data_dir, col, 0, num_rand_images)
    prepared_dataset = transform_dataset(dataset, feat_extract_transform)
    train_ds = prepared_dataset['train']
    val_ds = prepared_dataset['val']
    print(f'We have {len(train_ds)} training examples and {len(val_ds)} validation ones.')
    train_dataloader = DataLoader(train_ds, batch_size = batch_size, shuffle = False)
    val_dataloader = DataLoader(val_ds, batch_size = batch_size, shuffle = False)

    # Average loss
    average_loss = np.round(calculate_average_loss_over_dataset(model, base_device, train_dataloader), 3)

    save_config(data_dir, from_pretrained_model, model_checkpoint, model_arch,
                means, stand_devs, size, batch_size, base_device, average_loss, dir_to_save, title)
    
    end = time.time()

    print(f'This script took {np.round(end-start, 3)} seconds.')