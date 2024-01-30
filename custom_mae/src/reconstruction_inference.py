#########
# Imports
#########

# Standard
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# DL
from transformers import ViTMAEForPreTraining, ViTFeatureExtractor

###########
# Functions
###########

def load_vitmae_from_from_pretrained_w_weights(from_pretrained_model_path, weights_path):

    feature_extractor = ViTFeatureExtractor.from_pretrained(from_pretrained_model_path)

    if weights_path == 'None':
        print('We are loading in a pre-trained ViTMAE model.')
        model = ViTMAEForPreTraining.from_pretrained(from_pretrained_model_path)
    elif weights_path != 'None':
        print('We are loading in a pre-trained ViTMAE model and swapping out the weights')
        model = ViTMAEForPreTraining.from_pretrained(from_pretrained_model_path)
        checkpoint = torch.load(weights_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    elif weights_path != 'Scratch':
        print('We are loading in a ViTMAE pre-trained model, getting the config file, and re-initializing it.')
        model = ViTMAEForPreTraining.from_pretrained(from_pretrained_model_path)
        config_file = model.config
        model_from_config = ViTMAEForPreTraining._from_config(config_file)
        model = model_from_config

    return feature_extractor, model

def show_image(image, stds, means, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * stds + means) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def visualize(orig_image, pixel_values, model, stds, means, output_dir, title):
    # forward pass
    outputs = model(pixel_values)
    y = model.unpatchify(outputs.logits)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    
    # visualize the mask
    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', pixel_values)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 5, 1)
    plt.imshow(orig_image)
    plt.title('Original')

    plt.subplot(1, 5, 2)
    show_image(x[0], stds, means, "Prepared")

    plt.subplot(1, 5, 3)
    show_image(im_masked[0], stds, means, "Masked")

    plt.subplot(1, 5, 4)
    show_image(y[0], stds, means, "Reconstruction")

    plt.subplot(1, 5, 5)
    show_image(im_paste[0], stds, means, "Reconstruction + Visible")

    if output_dir:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the figures
        save_path = os.path.join(output_dir, title + '.png')
        plt.savefig(save_path)


############
# Running it
############

if __name__ == '__main__':

    # weights_path = '/sddata/projects/Cervical_Cancer_Projects/models/Facebook_Huge_Cervix_Custom_Finetune_v2_Cont_best_epoch.pth'
    weights_path = '/sddata/projects/SSL/custom_mae/Reconstruction_Custom_Finetuning_Best_Models/Facebook_Huge_Dia_Ret_Custom_Finetune_best_epoch.pth'
    feature_extractor, model = load_vitmae_from_from_pretrained_w_weights('facebook/vit-mae-huge', weights_path)
    print(f'\nModel prepared initial weights:')
    for name, param in model.named_parameters():
            print(f"Parameter: {name}")
            print(param.data)
            break

    # Loading 
    means = np.array(feature_extractor.image_mean)
    stds = np.array(feature_extractor.image_std)

    # Make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)

    image_paths = [
        # '/sddata/projects/Cervical_Cancer_Projects/data/full_dataset/full_dataset_duke_liger_itoju_5StLowQual/HFLD-00013476.jpg',
        #            '/sddata/projects/Cervical_Cancer_Projects/data/full_dataset/full_dataset_duke_liger_itoju_5StLowQual/I473982_C2.jpg',
        #            '/sddata/projects/Cervical_Cancer_Projects/data/full_dataset/full_dataset_duke_liger_itoju_5StLowQual/PA179.jpg',
        #            '/sddata/projects/Cervical_Cancer_Projects/data/SRA_IRIS_cambodia/MSK 000015 005.jpeg',
        #            '/sddata/projects/Cervical_Cancer_Projects/data/SRA_IRIS_cambodia/MSK 000034 002.jpeg',
        #            '/sddata/projects/Cervical_Cancer_Projects/data/SRA_IRIS_DR/DOM 000016 008.jpeg',
        #            '/sddata/projects/Cervical_Cancer_Projects/data/SRA_IRIS_DR/DOM 000030 012.jpeg',
                   '/sddata/data/retina_datasets/diabetic_retinopathy_detection/train/training/22711_left.jpeg',
                   '/sddata/data/retina_datasets/diabetic_retinopathy_detection/train/training/38569_left.jpeg',
                   '/sddata/data/retina_datasets/diabetic_retinopathy_detection/train/training/42189_right.jpeg'
                   ]
    titles = [
        # 'Full_Dataset_Cervix_example1', 'Full_Dataset_Cervix_example2', 'Full_Dataset_Cervix_example3', 
        #       'Cervix_Cambodia1', 'Cervix_Cambodia2', 'Dom_Rep_example1','Dom_Rep_example2', 
              'Dia_Ret_example1', 'Dia_Ret_example2', 'Dia_Ret_example3'
              ]
    # titles = ['ViTMAE_Huge_ImageNet_Finetuned_Plus_Domain_Full_Dataset_Cervix_Finetuning_' + title for title in titles]
    titles = ['ViTMAE_Huge_ImageNet_Finetuned_Plus_Domain_Dia_Ret_Finetuning_' + title for title in titles]
    output_dir = '/sddata/projects/SSL/custom_mae/Inference/Compilation/'
    for i in range(len(image_paths)):

        image = Image.open(image_paths[i])
        pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
        visualize(image, pixel_values, model, means, stds, output_dir, titles[i])