#!/bin/bash
#machine learning is distribution locked.
#do not expect this to work anywhere but ubuntu 22.04 lts
curl -LsSf https://astral.sh/uv/install.sh | sh
#astralsh documentation suggests curl ... | sh
source ~/.local/bin/env
uv python install 3.12
uv venv --seed --python 3.12

##vmuv-fasd-init.sh
#uv pip install math-verify[antlr4_13_2]
uv sync
source .venv/bin/activate
uv pip install flash-attn --no-build-isolation
uv pip install -e ".[all]"
##because we are fundamentally using willccbb's starting point there's
##a plausibly preexisting+working pyproj?