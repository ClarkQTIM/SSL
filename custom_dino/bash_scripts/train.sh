#!/bin/bash

python main_dino.py \
    --arch densenet121_pt \
    --optimizer sgd \
    --lr 0.03 \
    --weight_decay 1e-4 \
    --weight_decay_end 1e-4 \
    --global_crops_scale 0.14 1 \
    --local_crops_scale 0.05 0.14 \
    --data_path /sddata/projects/Cervical_Cancer_Projects/cervical_cancer_diagnosis/csvs/Beat_Model36/seed_nevada_no_pave.csv \
    --output_dir /sddata/projects/SSL/custom_dino/finetuning/PAVE/Portability_Comparison/DN121_seed_nevada_no_pave \
    --norm_mean 0.411 0.276 0.217 \
    --norm_std 0.235 0.195 0.185