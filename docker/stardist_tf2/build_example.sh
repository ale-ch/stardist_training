docker buildx build --platform linux/amd64 -t alech00/stardist_training:v0.1 --push .

singularity build stardist_training.sif docker://alech00/stardist_training:v0.1

