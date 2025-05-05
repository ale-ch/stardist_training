data_dir='/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/tests/test_imaging_data/data'
output_dir='/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/tests/test_imaging_data/data'
seed=42
scripts_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/stardist_training/scripts

python scripts_dir/split_image_data.py \
    --data_dir ${data_dir} \
    --output_dir ${output_dir} \
    --seed ${seed} \
    --split_ratio 0.8



