#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=aa100
#SBATCH --job-name=cclark_dino_cervix
#SBATCH --gres=gpu:1
#SBATCH --ntasks=10
#SBATCH --output=cclark_dino_cervix.%j.out
#SBATCH --error=cclark_dino_cervix%.err

module purge
module load anaconda
conda activate cclark_dl4

python main_dino.py \
    --arch dino_vitb16 \
    --optimizer sgd \
    --lr 0.03 \
    --weight_decay 1e-4 \
    --weight_decay_end 1e-4 \
    --global_crops_scale 0.14 1 \
    --local_crops_scale 0.05 0.14 \
    --data_path /projects/cclark@xsede.org/SSL/csvs/datasets/all_cervix_images_up_to_03262024_no_seed_test_test2_alpine.csv \
    --output_dir /projects/cclark@xsede.org/SSL/custom_dino/finetuning/cervix/Vitb16