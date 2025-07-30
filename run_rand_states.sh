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

# Log into wandb once
singularity exec \
    -B /hpcnfs \
    "${singularity_imgs_dir}/${training_image}" \
    wandb login

# Loop over random seeds
for random_seed in "${random_seeds[@]}"; do
    # Loop over epochs
    for epochs in "${epochs_list[@]}"; do
        model_name="stardist_e${epochs}_spe${steps_per_epoch}_lr${learning_rate}_vp${val_prop}_vpp${val_prop_opt}_rs${random_seed}"

        echo "Starting training with random seed: ${random_seed}, epochs: ${epochs}"
        echo "Model name: ${model_name}"

        #log_file="${cur_model_dir}/training_log.txt"
        #start_time=$(date "+%Y-%m-%d %H:%M:%S")
        #echo "start_time=$start_time" > "$log_file"

        if [ "$augment" = true ]; then
            singularity exec \
            -B /hpcnfs \
            "${singularity_imgs_dir}/${training_image}" \
            python "${scripts_dir}/main.py" \
                --base_dir ./ \
                --data_dir "${data_dir}" \
                --model_name "${model_name}" \
                --epochs "${epochs}" \
                --steps_per_epoch "${steps_per_epoch}" \
                --augment \
                --learning_rate "${learning_rate}" \
                --pretrained "${pretrained}" \
                --val_prop "${val_prop}" \
                --val_prop_opt "${val_prop_opt}" \
                --random_seed "${random_seed}"
        else
            singularity exec \
            -B /hpcnfs \
            "${singularity_imgs_dir}/${training_image}" \
            python "${scripts_dir}/main.py" \
                --base_dir ./ \
                --data_dir "${data_dir}" \
                --model_name "${model_name}" \
                --epochs "${epochs}" \
                --steps_per_epoch "${steps_per_epoch}" \
                --learning_rate "${learning_rate}" \
                --pretrained "${pretrained}" \
                --val_prop "${val_prop}" \
                --val_prop_opt "${val_prop_opt}" \
                --random_seed "${random_seed}"
        fi

        #end_time=$(date "+%Y-%m-%d %H:%M:%S")
        #echo "end_time=$end_time" >> "$log_file"

        # Convert model to TensorFlow 1 format
        echo "Exporting model for random seed: ${random_seed}, epochs: ${epochs}"
        singularity exec \
            -B /hpcnfs \
            "${singularity_imgs_dir}/${export_image}" \
            python "${scripts_dir}/export_model_tf1.py" \
                --base_dir "./runs/${random_seed}" \
                --model_name "${model_name}"
    done
done

echo "All training and export jobs completed"
