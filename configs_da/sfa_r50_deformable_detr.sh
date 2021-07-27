#!/usr/bin/env bash

set -x

EXP_DIR=exps_da/sfa_r50_dd_hda1_cmt
PY_ARGS=${@:1}

python -u main_da.py \
    --hda 1 --cmt \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
