singularity_imgs_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/singularity_images
base_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/tests/test_imaging_data
scripts_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/stardist_training
model_name=stardist

epochs=5
steps_per_epoch=5
val_prop=0.1

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
        --val_prop ${val_prop}

# Convert trained model to TensorFlow 1 format
singularity exec \
    -B /hpcnfs \
    ${singularity_imgs_dir}/stardist_export_tf1.sif \
    python ${scripts_dir}/export_model_tf1.py \
        --base_dir ${base_dir} \
        --model_name ${model_name}
