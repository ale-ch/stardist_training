# stardist_e100_spe50_vp0.1_vpp1  stardist_e150_spe50_vp0.1_vpp1  stardist_e200_spe50_vp0.1_vpp1  stardist_e50_spe50_vp0.1_vpp1  stardist_e80_spe50_vp0.1_vpp1


singularity_imgs_dir=/hpcnfs/scratch/DIMA/chiodin/singularity_images
scripts_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/stardist_training

#model_name=stardist_e80_spe50_vp0.1_vpp1

model_name=stardist_e25_spe50_lr0.0001_vp0.1_vpp1

run_name=20250505

gt_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/runs/${run_name}/data/test/masks

test_imgs_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/runs/${run_name}/data/test/images

preds_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/runs/${run_name}/predictions/${model_name}

models_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/runs/${run_name}/models

qc_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/runs/${run_name}/quality_control/${model_name}


mkdir -p ${preds_dir}
mkdir -p ${qc_dir}

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
        --preds_dir ${preds_dir} \
        --gt_dir ${gt_dir} \
        --outdir ${qc_dir}
