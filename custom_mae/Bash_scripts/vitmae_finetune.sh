MODEL_TYPE=facebook/vit-mae-huge
# MODEL_TYPE=None
MODEL_ARCH=None
MODEL_CHECKPOINT=None
# MODEL_ARCH=mae_vit_large_patch16
# MODEL_CHECKPOINT=/sddata/projects/SSL/custom_mae/Reconstruction_Custom_Finetuning_Best_Models/Facebook_Huge_Cervix_Custom_Finetune_v2_best_epoch.pth
# MEANS=0.485_0.456_0.406
# STAND_DEVS=0.229_0.224_0.225
# SIZE=224
MEANS=None
STAND_DEVS=None
SIZE=0
DATA=/sddata/projects/SSL/custom_mae/csvs/dr_training_images.csv
# DATA=/sddata/projects/SSL/custom_mae/csvs/all_cropped_fundus_images.csv
COL=Image
VAL_PCT=0.15
BATCH=16
EPOCHS=25
LR=6.25e-05
VIS=5
SAVE_DIR=//sddata/projects/SSL/custom_mae/Reconstruction_Custom_Finetuning_Outputs/Facebook_Huge_Dia_Ret_Custom_Finetune/
TITLE=Facebook_Huge_Dia_Ret_Custom_Finetune
GPUS=1
SEED=0

python3 /sddata/projects/SSL/custom_mae/src/vitmae_finetuner.py -model_type $MODEL_TYPE -model_arch $MODEL_ARCH -model_checkpt $MODEL_CHECKPOINT -means $MEANS -stand_devs $STAND_DEVS -size $SIZE -data $DATA -col $COL -val_pct $VAL_PCT -batch $BATCH -epochs $EPOCHS -lr $LR -vis $VIS -save_dir $SAVE_DIR -title $TITLE -gpus $GPUS -seed $SEED