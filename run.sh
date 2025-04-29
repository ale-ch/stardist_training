singularity exec -B /hpcnfs stardist_training.sif python /hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/stardist_training/main.py

singularity exec -B /hpcnfs ./singularity_images/stardist_export_tf1.sif python /hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/stardist_training/convert_model_tf1.p
