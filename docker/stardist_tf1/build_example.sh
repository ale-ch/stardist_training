docker buildx build --platform linux/amd64 -t alech00/stardist_export_tf1:v0.0 --push .

singularity build stardist_export_tf1.sif docker://alech00/stardist_export_tf1:v0.0

singularity exec -B /hpcnfs ./singularity_images/stardist_export_tf1.sif python /hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/stardist_training/convert_model_tf1.py
