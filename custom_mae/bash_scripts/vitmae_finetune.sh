MODEL_TYPE=facebook/vit-mae-base
MODEL_ARCH=None
MODEL_CHECKPOINT=/sddata/projects/SSL/custom_mae/Reconstruction_Custom_Finetuning_Best_Models/ViTMAE_Base_Cervix_all_data_but_test_test2_cont_best_epoch.pth
MEANS=0.411_0.276_0.217
STAND_DEVS=0.236_0.195_0.185
SIZE=224
# MEANS=None
# STAND_DEVS=None
# SIZE=0
DATA=/sddata/projects/SSL/csvs/datasets/all_cervix_images_up_to_03262024_no_seed_test_test2.csv
COL=image
VAL_PCT=0.15
BATCH=16
EPOCHS=29
LR=6.25e-05
VIS=5
SAVE_DIR=/sddata/projects/SSL/custom_mae/Reconstruction_Custom_Finetuning_Outputs/Cervix_04122024_cont_cont_/ViTMAE_Base_Cervix_all_data_no_test_test2/
TITLE=ViTMAE_Base_Cervix_all_data_but_test_test2_cont_cont
GPUS=0
SEED=0

python3 /sddata/projects/SSL/custom_mae/src/vitmae_finetuner.py -model_type $MODEL_TYPE -model_arch $MODEL_ARCH -model_checkpt $MODEL_CHECKPOINT -means $MEANS -stand_devs $STAND_DEVS -size $SIZE -data $DATA -col $COL -val_pct $VAL_PCT -batch $BATCH -epochs $EPOCHS -lr $LR -vis $VIS -save_dir $SAVE_DIR -title $TITLE -gpus $GPUS -seed $SEED