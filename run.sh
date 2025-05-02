singularity_imgs_dir=/hpcnfs/scratch/DIMA/chiodin/singularity_images


singularity exec \
    -B /hpcnfs \
    ${singularity_imgs_dir}/stardist_training.sif \
    wandb login


scripts_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/stardist_training
epochs=5
steps_per_epoch=5
val_prop=0.1
val_prop_opt=1
augment=true
random_seed=42

model_name=stardist_e${epochs}_spe${steps_per_epoch}_vp${val_prop}_vpp${val_prop_opt}
models_dir=./models
cur_model_dir=./models/${model_name}
gt_dir=./data/test/masks
test_imgs_dir=./data/test/images
preds_dir=./predictions
qc_dir=./quality_control

mkdir -p ${cur_model_dir}

# Save to text file
echo "Saving training configuration to ${cur_model_dir}/training_conf.txt"
cat <<EOF > ${cur_model_dir}/training_conf.txt
model_name=$model_name
epochs=$epochs
steps_per_epoch=$steps_per_epoch
augment=${augment}
val_prop=$val_prop
val_prop_opt=$val_prop_opt
EOF


# File to write times to
log_file="${cur_model_dir}/training_log.txt"


# Record start time
start_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "start_time=$start_time" > "$log_file"

if [ "$augment" = true ]; then
    # Run training
    singularity exec \
    -B /hpcnfs \
    ${singularity_imgs_dir}/stardist_training.sif \
    python ${scripts_dir}/main.py \
        --base_dir ./ \
        --model_name ${model_name} \
        --epochs ${epochs} \
        --steps_per_epoch ${steps_per_epoch} \
        --augment \
        --val_prop ${val_prop} \
        --val_prop_opt ${val_prop_opt} \
        --random_seed ${random_seed}
else
    # Run training
    singularity exec \
    -B /hpcnfs \
    ${singularity_imgs_dir}/stardist_training.sif \
    python ${scripts_dir}/main.py \
        --base_dir ./ \
        --model_name ${model_name} \
        --epochs ${epochs} \
        --steps_per_epoch ${steps_per_epoch} \
        --val_prop ${val_prop} \
        --val_prop_opt ${val_prop_opt}
        --random_seed ${random_seed}
fi


# Record end time
end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "end_time=$end_time" >> "$log_file"



# Convert trained model to TensorFlow 1 format
singularity exec \
    -B /hpcnfs \
    ${singularity_imgs_dir}/stardist_export_tf1.sif \
    python ${scripts_dir}/export_model_tf1.py \
        --base_dir ./ \
        --model_name ${model_name}
