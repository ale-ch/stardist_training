#!/bin/bash

# Configuration variables
singularity_imgs_dir=/hpcnfs/scratch/DIMA/chiodin/singularity_images
training_image=stardist_training_v02.sif
export_image=stardist_export_tf1.sif

scripts_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/stardist_training
data_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/data

# Training parameters
steps_per_epoch=2
val_prop=0.1
val_prop_opt=1
augment=true
learning_rate=0.0001
pretrained=2D_versatile_fluo

# Directory paths
models_dir=./models
gt_dir=./data/test/masks
test_imgs_dir=./data/test/images
preds_dir=./predictions
qc_dir=./quality_control

# Define arrays for seeds and epochs
random_seeds=(42 51 62 71 80)
epochs_list=(200 300)  # You can add more epochs like (2 5 10) if needed

singularity exec \
    -B /hpcnfs \
    "${singularity_imgs_dir}/${training_image}" \
    python ${scripts_dir}/main.py \
        --base_dir ./ \
        --data_dir ${data_dir} \
        --pretrained 2D_versatile_fluo \
        --test_prop 0.1 \
        --val_prop 0.1 \
        --val_prop_opt 1.0 \
        --random_seeds 1238 \
        --epochs_list 10 \
        --steps_list 2 \
        --lr_list 0.0001 \
        --train_reduce_lr '{"factor": 0.5, "patience": 5, "min_delta": 0.0001}' '{"factor": 0.1, "patience": 10, "min_delta": 0.0002}' \
        --augment_list True \
        --early_stopping_list '{"monitor": "val_prob_loss", "min_delta": 0.1, "patience": 0, "verbose": 0, "baseline": null, "restore_best_weights": false, "start_from_epoch": 0, "mode": "min"}'

