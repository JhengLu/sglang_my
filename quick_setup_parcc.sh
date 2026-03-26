set -e

if [ -z "$HF_MODELS" ]; then
    export HF_MODELS=/vast/projects/liuv/pennnetworks/hf_models
    if ! grep -q 'export HF_MODELS=/vast/projects/liuv/pennnetworks/hf_models' ~/.bashrc; then
        echo 'export HF_MODELS=/vast/projects/liuv/pennnetworks/hf_models' >> ~/.bashrc
    fi
fi
echo $HF_MODELS
cd $proj_home/sglang_my

# Create conda env and install packages
eval "$(conda shell.bash hook)"
conda create -n sglang python=3.12 -y
conda install -n sglang cuda-toolkit -c nvidia -y
conda activate sglang
export CUDA_HOME=$CONDA_PREFIX

if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$proj_home/.local/bin sh

    source $proj_home/.local/bin/env
else
    echo "uv is already installed, skipping installation."
fi

uv pip install --upgrade pip
uv pip install -e "python"
