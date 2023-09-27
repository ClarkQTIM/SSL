#!/usr/bin/bash
MODEL=res101
CHECKPOINT=None
IMG_DIR=csvs/cifar100_paths_labels.csv
BATCH_SIZE=64
EPOCHS=50
LEARNING_RATE=1e-3
AUGMENTATIONS=py_files/augs.json
DIR=Results/Cifar100/Res101/
TITLE=Classification_Only_Res101
GPUS=1
SEED=1234
CUTOFF=0 # Testing on a smaller dataset if cutoff != 0
python3 py_files/full_train_downstream.py -m $MODEL -k $CHECKPOINT -i $IMG_DIR -t $TRAIN_PERCENT -b $BATCH_SIZE -e $EPOCHS -r $LEARNING_RATE -a $AUGMENTATIONS -d $DIR -s $TITLE -p -v -g $GPUS --seed $SEED -c $CUTOFF