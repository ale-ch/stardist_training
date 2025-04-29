# Dockerfile for TensorFlow 1.x Model Export Environment
FROM continuumio/miniconda3:latest

# Create and set up Conda environment
RUN conda create -y -n tf1_model_export python=3.7 && \
    echo "conda activate tf1_model_export" >> ~/.bashrc

# Install packages
RUN /opt/conda/envs/tf1_model_export/bin/conda install -y pip && \
    /opt/conda/envs/tf1_model_export/bin/pip install \
    "tensorflow<2" \
    "csbdeep[tf1]" \
    "stardist[tf1]"

# Set environment variables
ENV PATH /opt/conda/envs/tf1_model_export/bin:$PATH

# Set working directory
WORKDIR /workspace

# Verify installation
RUN python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" && \
    python -c "import stardist; print(f'StarDist version: {stardist.__version__}')"

# Default command (can override when running)
CMD ["python", "--version"]
