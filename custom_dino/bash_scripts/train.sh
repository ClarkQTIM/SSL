#!/bin/bash

python main_dino.py \
    --arch densenet121_pt \
    --optimizer sgd \
    --lr 0.03 \
    --weight_decay 1e-4 \
    --weight_decay_end 1e-4 \
    --global_crops_scale 0.14 1 \
    --local_crops_scale 0.05 0.14 \
    --data_path /sddata/projects/SSL/csvs/datasets/all_cervix_images_up_to_04122024_no_seed_test_test2.csv \
    --output_dir /sddata/projects/SSL/custom_dino/finetuning/newest_runs_04062024/cervix/DenseNet121