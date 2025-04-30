singularity_imgs_dir=/hpcnfs/scratch/DIMA/chiodin/singularity_images
base_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/runs/
scripts_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/stardist_training

epochs=5
steps_per_epoch=5
val_prop=0.1
val_prop_opt=1
augment=true


model_name=stardist_e${epochs}_spe${steps_per_epoch}_vp${val_prop}_vpp${val_prop_opt}
model_dir=./models/${model_name}

mkdir -p ${model_dir}

# Save to text file
echo "Saving training configuration to ${model_dir}/training_conf.txt"
cat <<EOF > ${model_dir}/training_conf.txt
model_name=$model_name
epochs=$epochs
steps_per_epoch=$steps_per_epoch
augment=${augment}
val_prop=$val_prop
val_prop_opt=$val_prop_opt
EOF


# File to write times to
log_file="${model_dir}/training_log.txt"

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
        --val_prop_opt ${val_prop_opt}
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
fi



## Run training
#singularity exec \
#    -B /hpcnfs \
#    ${singularity_imgs_dir}/stardist_training.sif \
#    python ${scripts_dir}/main.py \
#        --base_dir ./ \
#        --model_name ${model_name} \
#        --epochs ${epochs} \
#        --steps_per_epoch ${steps_per_epoch} \
#        --augment \
#        --val_prop ${val_prop} \
#        --val_prop_opt ${val_prop_opt}

# Convert trained model to TensorFlow 1 format
singularity exec \
    -B /hpcnfs \
    ${singularity_imgs_dir}/stardist_export_tf1.sif \
    python ${scripts_dir}/export_model_tf1.py \
        --base_dir ./ \
        --model_name ${model_name}

# Record end time
end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "end_time=$end_time" >> "$log_file"
