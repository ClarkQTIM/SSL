#########
# Imports
#########

# Standard
import os
import json
import time

# DL
from monai.data import DataLoader

# Py Files
from utils import *

###########
# Functions
###########

def calculate_average_loss_over_dataset(model, base_device, prepared_dataset, batch_size):

    dataloader = DataLoader([prepped_data.squeeze(0) for prepped_data in prepared_dataset], batch_size = batch_size, shuffle = False)

    overall_loss = 0

    for batch in dataloader:

        outputs = model(batch.to(base_device))
        try:
            loss_on_batch = outputs['loss']
        except:
            loss_on_batch = outputs[0]

        overall_loss += loss_on_batch.detach().cpu()

    return overall_loss/len(dataloader)

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

    # data_dir = '/sddata/projects/data/full_dataset_duke_liger_itoju_5StLowQual'
    data_dir = '/sddata/projects/SSL/custom_mae/csvs/all_cropped_fundus_images.csv'
    from_pretrained_model = "facebook/vit-mae-huge"
    # model_checkpoint = 'sddata/projects/Cervical_Cancer_Projects/SSL/custom_mae/Reconstruction_Custom_Finetuning_Best_Models/Facebook_Huge_Cervix_Custom_Finetune_best_epoch.pth'
    # model_checkpoint =  '/sddata/projects/SSL/custom_mae/Reconstruction_Custom_Finetuning_Best_Models/Facebook_Huge_Cropped_Fundus_Custom_Finetune_best_epoch.pth'
    # model_arch = 'mae_vit_large_patch16'
    model_checkpoint = None
    model_arch = None
    batch_size = 32
    title = 'Facebook_Huge_all_cropped_fundus_images_Average_Loss'
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

    start = time.time()

    print(data_dir, from_pretrained_model, model_checkpoint, model_arch,
          means, stand_devs, size, batch_size, dir_to_save, title)

    if from_pretrained_model != None and model_checkpoint == None:
        print('We are loading in the model with from_pretrained.')

        feature_extractor, model = load_vitmae_from_from_pretrained(from_pretrained_model)
        base_device = configure_devices(model, gpus)
        prepared_dataset, valid_image_paths = prepare_dataset_from_dir_parallel(data_dir, 'Image', feature_extractor, None, None, None, None)
        average_loss = np.round(calculate_average_loss_over_dataset(model, base_device, prepared_dataset, batch_size), 3)

    elif model_arch != None and model_checkpoint != None:
        print('We are loading in the model from an architecture and weights.')

        model = load_vitmae_from_arch_weights(model_arch, model_checkpoint)
        base_device = configure_devices(model, gpus)
        prepared_dataset, valid_image_paths = prepare_dataset_from_dir_parallel(data_dir, 'Image', None, size, means, stand_devs, None)
        average_loss = np.round(calculate_average_loss_over_dataset(model, base_device, prepared_dataset, batch_size), 3)

    elif from_pretrained_model != None and model_checkpoint != None:
        print('We are loading in the model architecture from from_pretrained and also replacing the weights.')

        feature_extractor, model = load_vitmae_from_from_pretrained_w_weights(from_pretrained_model, model_checkpoint)
        base_device = configure_devices(model, gpus)
        prepared_dataset, valid_image_paths = prepare_dataset_from_dir_parallel(data_dir, 'Image', feature_extractor, None, None, None, None)
        average_loss = np.round(calculate_average_loss_over_dataset(model, base_device, prepared_dataset, batch_size), 3)

    else:
        print(f'There is an issue somewhere. The from_pretrained is {from_pretrained_model}, the architecture is {model_arch}, and the checkpoint is {model_checkpoint}.')

    save_config(data_dir, from_pretrained_model, model_checkpoint, model_arch,
                means, stand_devs, size, batch_size, base_device, average_loss, dir_to_save, title)
    
    end = time.time()

    print(f'This script took {np.round(end-start, 3)} seconds.')

    




