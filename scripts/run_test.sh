#!/bin/bash

python -m compert.train --dataset_path ../cpa_binaries/datasets/GSM_new.h5ad --save_dir /tmp --max_epochs 1  --doser_type sigm
