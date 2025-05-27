singularity_imgs_dir=/hpcnfs/scratch/DIMA/chiodin/singularity_images
scripts_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/stardist_training
#model_name=stardist_full_e200_lr00001_aug1_seed10_es50p0.001_rlr0.5p50
models_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/runs/20250514/runs/10/models
model_name=stardist_fixed_full_e200_lr00001_aug1_seed10_es200p0.001_rlr0.5p200
test_imgs_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/runs/20250514/test_images
outdir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/runs/20250514/predictions

singularity exec \
    -B /hpcnfs \
    ${singularity_imgs_dir}/stardist_training_v02.sif \
    python ${scripts_dir}/scripts/predict2.py \
        --model_name ${model_name} \
        --models_dir ${models_dir} \
        --test_imgs_dir ${test_imgs_dir} \
        --outdir ${outdir}
