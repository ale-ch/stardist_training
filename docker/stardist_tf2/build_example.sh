docker buildx build --platform linux/amd64 -t alech00/stardist_training:v0.0 --push .

singularity build stardist_training.sif docker://alech00/stardist_training:v0.0

singularity exec -B /hpcnfs stardist_training.sif python /hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/stardist_training/main.py
