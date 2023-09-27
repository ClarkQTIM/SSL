#########
# Imports
#########

# Standard
import numpy as np

# Py Files
from utils import *


if __name__ == "__main__":

    # data_dir = '/projects/Cervical_Cancer_Projects/data/full_dataset_duke_liger_itoju_5StLowQual'
    data_dir = '/projects/Fundus_Segmentation/Organized_Datasets/Rim_One_r1_Organized/Images'
    save_dir = '/projects/Cervical_Cancer_Projects/SSL/custom_mae/Inference/ViTMAE_Fundus/Facebook_Base/'

    # model_checkpoint = '/projects/Cervical_Cancer_Projects/SSL/mae_facebook/demo/mae_visualize_vit_large_ganloss.pth'
    # model_arch = 'mae_vit_large_patch16'
    # means = np.array([0.485, 0.456, 0.406])
    # stand_devs = np.array([0.229, 0.224, 0.225])
    # size = 224
    # model = load_vitmae_from_arch_weights(model_arch, model_checkpoint)
    # feature_extractor = (means, stand_devs)
    # prepared_dataset, valid_image_paths = prepare_dataset_from_dir(data_dir, None, size, means, stand_devs, 10)

    from_pretrained_model = 'facebook/vit-mae-base'
    feature_extractor, model = load_vitmae_from_from_pretrained(from_pretrained_model)
    prepared_dataset, valid_image_paths = prepare_dataset_from_dir_parallel(data_dir, feature_extractor, None, None, None, 10)

    # from_pretrained_model = 'facebook/vit-mae-huge'
    # model_checkpoint = '/projects/Cervical_Cancer_Projects/SSL/custom_mae/Reconstruction_Finetune_Best_Models/Facebook_Huge_Cervix_Custom_Finetune_best_epoch.pth'
    # feature_extractor, model = load_vitmae_from_from_pretrained_w_weights(from_pretrained_model, model_checkpoint)
    # prepared_dataset, valid_image_paths = prepare_dataset_from_dir_parallel(data_dir, feature_extractor, None, None, None, 10)

    for i in range(len(prepared_dataset)):

        visualize(None, prepared_dataset[i], model, feature_extractor, False, True, save_dir, valid_image_paths[i].split('/')[-1].split('.')[0])