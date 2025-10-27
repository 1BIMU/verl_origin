#!/usr/bin/env bash
set -uxo pipefail

export TRAIN_FILE=${TRAIN_FILE:-"data/dapo-math-17k.parquet"}
export OVERWRITE=${OVERWRITE:-1}


if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
fi

python transform_dapo_prompt.py