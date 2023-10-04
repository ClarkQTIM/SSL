MODEL_TYPE=facebook/vit-mae-huge
# MODEL_TYPE=None
MODEL_ARCH=None
MODEL_CHECKPOINT=/sddata/projects/SSL/custom_mae/Reconstruction_Custom_Finetuning_Best_Models/Facebook_Huge_Cervix_Custom_Finetune_v2_best_epoch.pth
# MODEL_ARCH=mae_vit_large_patch16
# MODEL_CHECKPOINT=/projects/Cervical_Cancer_Projects/SSL/mae_facebook/demo/mae_visualize_vit_large_ganloss.pth
# MEANS=0.485_0.456_0.406
# STAND_DEVS=0.229_0.224_0.225
# SIZE=224
MEANS=None
STAND_DEVS=None
SIZE=0
# DATA=/projects/Cervical_Cancer_Projects/data/full_dataset_duke_liger_itoju_5StLowQual
DATA_DIR='/sddata/projects/Cervical_Cancer_Projects/data/full_dataset_duke_liger_itoju_5StLowQual'
DATA_CSV='/sddata/projects/Cervical_Cancer_Projects/data/full_dataset_duke_liger_split_df.csv'
IMAGE_COL=MASKED_IMG_ID
LABEL_COL=CC_ST
VAL_PCT=0
BATCH=16
EPOCHS=25
LR=6.25e-05
VIS=5
SAVE_DIR=/sddata/projects/SSL/custom_mae/Downstream_Classification_Finetuning_Outputs/Facebook_Huge_full_dataset_liger_Finetuned/
TITLE=Facebook_Huge_full_dataset_liger_Finetuned
GPUS=0
SEED=0

python3 /sddata/projects/SSL/custom_mae/src/vitmae_downstream_classifier.py -model_type $MODEL_TYPE -model_arch $MODEL_ARCH -model_checkpt $MODEL_CHECKPOINT -means $MEANS -stand_devs $STAND_DEVS -size $SIZE -data_dir $DATA_DIR -data_csv $DATA_CSV -image_col $IMAGE_COL -label_col $LABEL_COL -val_pct $VAL_PCT -batch $BATCH -epochs $EPOCHS -lr $LR -vis $VIS -save_dir $SAVE_DIR -title $TITLE -gpus $GPUS -seed $SEED