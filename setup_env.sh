# NOTE: this will install mamba under /notebooks
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p /notebooks/miniconda
source /notebooks/miniconda/bin/activate
