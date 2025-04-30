singularity_imgs_dir=/hpcnfs/scratch/DIMA/chiodin/singularity_images
base_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/tests/test_imaging_data
scripts_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/stardist_training
model_name=stardist
model_dir=${base_dir}/models/${model_name}

epochs=5
steps_per_epoch=5
val_prop=0.1
val_prop_opt=1

# Save to text file
echo "Saving training configuration to ${model_dir}/training_conf.txt"
cat <<EOF > ${model_dir}/training_conf.txt
model_name=$model_name
epochs=$epochs
steps_per_epoch=$steps_per_epoch
val_prop=$val_prop
val_prop_opt=$val_prop_opt
EOF



# Run training
singularity exec \
    -B /hpcnfs \
    ${singularity_imgs_dir}/stardist_training.sif \
    python ${scripts_dir}/main.py \
        --base_dir ${base_dir} \
        --model_name ${model_name} \
        --epochs ${epochs} \
        --steps_per_epoch ${steps_per_epoch} \
        --augment \
        --val_prop ${val_prop} \
        --val_prop_opt ${val_prop_opt}

# Convert trained model to TensorFlow 1 format
singularity exec \
    -B /hpcnfs \
    ${singularity_imgs_dir}/stardist_export_tf1.sif \
    python ${scripts_dir}/export_model_tf1.py \
        --base_dir ${base_dir} \
        --model_name ${model_name}
