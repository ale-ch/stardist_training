#conda create -y --name tf1_model_export python=3.7
#conda activate tf1_model_export
conda install pip
pip install "tensorflow<2"
pip install "csbdeep[tf1]"
pip install "stardist[tf1]"
#conda deactivate
