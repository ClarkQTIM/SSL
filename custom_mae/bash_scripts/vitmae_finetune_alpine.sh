#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=aa100
#SBATCH --job-name=cclark_dino_cervix
#SBATCH --gres=gpu:1
#SBATCH --ntasks=10
#SBATCH --output=cclark_vitmae_cervix.%j.out
#SBATCH --error=cclark_vitmae_cervix%.err

module purge
module load anaconda
conda activate cclark_dl4

MODEL_TYPE=facebook/vit-mae-base
MODEL_ARCH=None
MODEL_CHECKPOINT=None
MEANS=0.411_0.276_0.217
STAND_DEVS=0.236_0.195_0.185
SIZE=224
# MEANS=None
# STAND_DEVS=None
# SIZE=0
DATA=/projects/cclark@xsede.org/SSL/csvs/datasets/all_cervix_images_up_to_03262024_no_seed_test_test2_alpine.csv
COL=image
VAL_PCT=0.15
BATCH=16
EPOCHS=50
LR=6.25e-05
VIS=5
SAVE_DIR=/projects/cclark@xsede.org/SSL/custom_mae/Reconstruction_Custom_Finetuning_Outputs/Cervix_04122024/ViTMAE_Base_Cervix_all_data_no_test_test2/
TITLE=ViTMAE_Base_Cervix_all_data_but_test_test2
GPUS=0
SEED=0

python3 /projects/cclark@xsede.org/SSL/custom_mae/src/vitmae_finetuner.py -model_type $MODEL_TYPE -model_arch $MODEL_ARCH -model_checkpt $MODEL_CHECKPOINT -means $MEANS -stand_devs $STAND_DEVS -size $SIZE -data $DATA -col $COL -val_pct $VAL_PCT -batch $BATCH -epochs $EPOCHS -lr $LR -vis $VIS -save_dir $SAVE_DIR -title $TITLE -gpus $GPUS -seed $SEED