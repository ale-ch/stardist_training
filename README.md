# Stardist Training Pipeline

## Files and Directories

**Scripts:**
- `scripts/` - Scripts to test/use individual components of the pipeline
- `utils/` - Utility functions for the pipeline
- `main.py` - Main script to run the Stardist training pipeline
- `export_model_tf1.py` - Script to convert Stardist models to TensorFlow 1 format (to be used in Imagej Fiji)

## Run Options

- `run_conda.sh` - Run using conda environment
- `run_singularity.sh` - Run using a singularity image built from the Dockerfile
- `run_rand_states.sh` - Run multiple random states and convert model to tf1 using singularity image

## Configuration

- `config.yaml` - Stardist model configuration file parameters

## Docker Images

- `stardist_tf1` - Image for converting stardist models to tensorflow 1
- `stardist_tf2` - Image for running training and inference with stardist models
