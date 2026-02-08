if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
else
    echo "uv is already installed, skipping installation."
fi

if ! grep -q 'export HF_MODELS=/data/hf_models' ~/.bashrc; then
    echo 'export HF_MODELS=/data/hf_models' >> ~/.bashrc
fi
export HF_MODELS=/data/hf_models
echo $HF_MODELS
cd ~/sglang_my

# Install the python packages
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install -e "python"