#!/bin/bash

python main_dino.py \
    --arch dino_vitb16 \
    --optimizer sgd \
    --lr 0.03 \
    --weight_decay 1e-4 \
    --weight_decay_end 1e-4 \
    --global_crops_scale 0.14 1 \
    --local_crops_scale 0.05 0.14 \
    --data_path /sddata/projects/SSL/csvs/datasets/all_dr_images_no_test.csv \
    --output_dir /sddata/projects/SSL/dino/finetuning/testing