docker buildx build --platform linux/amd64 -t alech00/stardist_export_tf1:v0.0 --push .

singularity build stardist_export_tf1.sif docker://alech00/stardist_export_tf1:v0.0
