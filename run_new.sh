singularity_imgs_dir=/hpcnfs/scratch/DIMA/chiodin/singularity_images
scripts_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/stardist_training
steps_per_epoch=10
val_prop=0.1
val_prop_opt=1
augment=true
random_seed=42
learning_rate=0.0001
models_dir=./models
gt_dir=./data/test/masks
test_imgs_dir=./data/test/images
preds_dir=./predictions
qc_dir=./quality_control

# Log into wandb once
singularity exec \
    -B /hpcnfs \
    ${singularity_imgs_dir}/stardist_training.sif \
    wandb login

# Loop o’er epochs
for epochs in 200; do

    model_name=stardist_e${epochs}_spe${steps_per_epoch}_lr${learning_rate}_vp${val_prop}_vpp${val_prop_opt}
    cur_model_dir=${models_dir}/${model_name}
    mkdir -p ${cur_model_dir}

    log_file="${cur_model_dir}/training_log.txt"
    start_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo "start_time=$start_time" > "$log_file"

    if [ "$augment" = true ]; then
        singularity exec \
        -B /hpcnfs \
        ${singularity_imgs_dir}/stardist_training.sif \
        python ${scripts_dir}/main.py \
            --base_dir ./ \
            --model_name ${model_name} \
            --epochs ${epochs} \
            --steps_per_epoch ${steps_per_epoch} \
            --augment \
            --learning_rate ${learning_rate} \
            --val_prop ${val_prop} \
            --val_prop_opt ${val_prop_opt} \
            --random_seed ${random_seed}
    else
        singularity exec \
        -B /hpcnfs \
        ${singularity_imgs_dir}/stardist_training.sif \
        python ${scripts_dir}/main.py \
            --base_dir ./ \
            --model_name ${model_name} \
            --epochs ${epochs} \
            --steps_per_epoch ${steps_per_epoch} \
            --learning_rate ${learning_rate}
            --val_prop ${val_prop} \
            --val_prop_opt ${val_prop_opt} \
            --random_seed ${random_seed}
    fi

    end_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo "end_time=$end_time" >> "$log_file"

    # Convert model to TensorFlow 1 format
    singularity exec \
        -B /hpcnfs \
        ${singularity_imgs_dir}/stardist_export_tf1.sif \
        python ${scripts_dir}/export_model_tf1.py \
            --base_dir ./ \
            --model_name ${model_name}

done

(base) [ieo7086@cn14 7436519]$ singularity_imgs_dir=/hpcnfs/scratch/DIMA/chiodin/singularity_images
(base) [ieo7086@cn14 7436519]$ scripts_dir=/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/stardist_training
(base) [ieo7086@cn14 7436519]$ steps_per_epoch=10
(base) [ieo7086@cn14 7436519]$ val_prop=0.1
(base) [ieo7086@cn14 7436519]$ val_prop_opt=1
(base) [ieo7086@cn14 7436519]$ augment=true
(base) [ieo7086@cn14 7436519]$ random_seed=42
(base) [ieo7086@cn14 7436519]$ learning_rate=0.0001
(base) [ieo7086@cn14 7436519]$ models_dir=./models
(base) [ieo7086@cn14 7436519]$ gt_dir=./data/test/masks
(base) [ieo7086@cn14 7436519]$ test_imgs_dir=./data/test/images
(base) [ieo7086@cn14 7436519]$ preds_dir=./predictions
(base) [ieo7086@cn14 7436519]$ qc_dir=./quality_control
(base) [ieo7086@cn14 7436519]$ 
(base) [ieo7086@cn14 7436519]$ # Log into wandb once
(base) [ieo7086@cn14 7436519]$ singularity exec \
>     -B /hpcnfs \
>     ${singularity_imgs_dir}/stardist_training.sif \
>     wandb login
=$val_prop_opt
random_seed=$random_seed
EOF

    log_file="${cur_model_dir}/training_log.txt"
    start_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo "start_time=$start_time" > "$log_file"

    if [ "$augment" = true ]; then
        singularity exec \
        -B /hpcnfs \
        ${singularity_imgs_dir}/stardist_training.sif \
        python ${scripts_dir}/main.py \
            --base_dir ./ \
            --model_name ${model_name} \
            --epochs ${epochs} \
            --steps_per_epoch ${steps_per_epoch} \
            --augment \
            --learning_rate ${learning_rate} \
            --val_prop ${val_prop} \
            --val_prop_opt ${val_prop_opt} \
            --random_seed ${random_seed}
    else
        singularity exec \
        -B /hpcnfs \
        ${singularity_imgs_dir}/stardist_training.sif \
        python ${scripts_dir}/main.py \
            --base_dir ./ \
            --model_name ${model_name} \
            --epochs ${epochs} \
            --steps_per_epoch ${steps_per_epoch} \
            --learning_rate ${learning_rate}
            --val_prop ${val_prop} \
            --val_prop_opt ${val_prop_opt} \
            --random_seed ${random_seed}
    fi

    end_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo "end_time=$end_time" >> "$log_file"

    # Convert model to TensorFlow 1 format
    singularity exec \
        -B /hpcnfs \
        ${singularity_imgs_dir}/stardist_export_tf1.sif \
        python ${scripts_dir}/export_model_tf1.py \
            --base_dir ./ \
            --model_name ${model_name}

done