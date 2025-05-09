#!/bin/bash

# Configuration variables
singularity_imgs_dir=/hpcnfs/scratch/DIMA/chiodin/singularity_images
training_image=stardist_training_v02.sif

scripts_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/stardist_training
data_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/data
base_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/
config_file=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/runs/20250507/config.yaml

singularity exec \
    -B /hpcnfs \
    "${singularity_imgs_dir}/${training_image}" \
    python ${scripts_dir}/main.py \
        --base_dir ${base_dir} \
        --data_dir ${data_dir} \
        --config_file ${config_file}

