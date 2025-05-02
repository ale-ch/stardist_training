singularity_imgs_dir=/hpcnfs/scratch/DIMA/chiodin/singularity_images
scripts_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/stardist_training

model_name=stardist_e200_spe50_vp0.1_vpp1
gt_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/runs/20250430/data/test/masks
test_imgs_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/runs/20250430/data/test/images
preds_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/predictions

models_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/runs/20250430/models


singularity exec \
    -B /hpcnfs \
    ${singularity_imgs_dir}/stardist_training.sif \
    python ${scripts_dir}/scripts/predict.py \
        --model_name ${model_name} \
        --models_dir ${models_dir} \
        --test_imgs_dir ${test_imgs_dir} \
        --outdir ${preds_dir}




singularity exec \
    -B /hpcnfs \
    ${singularity_imgs_dir}/stardist_training.sif \
    python ${scripts_dir}/scripts/quality_control.py \
        --preds_dir /hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/predictions/${model_name} \
        --gt_dir ${gt_dir} \
        --outdir /hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/quality_control/${model_name}
