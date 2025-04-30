singularity_imgs_dir=/hpcnfs/scratch/DIMA/chiodin/singularity_images
scripts_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/stardist_training

model_name=stardist_e80_spe25_vp0.1_vpp1
base_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/runs/20250430/models
file=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/runs/20250430/data/test/images/dapi_28.tif
outdir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/predictions
indir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/runs/20250430/data/test/masks

singularity exec \
    -B /hpcnfs \
    ${singularity_imgs_dir}/stardist_training.sif \
    python ${scripts_dir}/scripts/predict.py \
        --model_name ${model_name} \
        --base_dir ${base_dir} \
        --indir ${indir} \
        --outdir ${outdir}
