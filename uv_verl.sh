set -ex
# curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv uv_verl --python 3.10
source uv_verl/bin/activate
export UV_LINK_MODE=copy
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
# torch==2.6.0 transfomers==4.51.1 vlm==0.8.5.post1 sglang==0.4.6.post1 peft==0.15.2
uv pip install --no-deps -e .
uv pip install --upgrade setuptools
uv pip install math-verify[antlr4_9_3]