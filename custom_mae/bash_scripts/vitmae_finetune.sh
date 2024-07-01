MODEL_TYPE=facebook/vit-mae-base
MODEL_ARCH=None
MODEL_CHECKPOINT=None
MEANS=0.313_0.222_0.164
STAND_DEVS=0.306_0.222_0.176
SIZE=224
# MEANS=None
# STAND_DEVS=None
# SIZE=0
DATA=/sddata/projects/SSL/csvs/datasets/DR/1_DR_ssl_frac.csv
COL=image
VAL_PCT=0.15
BATCH=16
EPOCHS=50
LR=6.25e-05
VIS=5
SAVE_DIR=/sddata/projects/SSL/custom_mae/Reconstruction_Custom_Finetuning_Outputs/DR_06062024/frac1/
TITLE=ViTMAE_Base_DR_all_data_no_test_1
GPUS=1
SEED=0

python3 /sddata/projects/SSL/custom_mae/src/vitmae_finetuner.py -model_type $MODEL_TYPE -model_arch $MODEL_ARCH -model_checkpt $MODEL_CHECKPOINT -means $MEANS -stand_devs $STAND_DEVS -size $SIZE -data $DATA -col $COL -val_pct $VAL_PCT -batch $BATCH -epochs $EPOCHS -lr $LR -vis $VIS -save_dir $SAVE_DIR -title $TITLE -gpus $GPUS -seed $SEED