#!/bin/bash

python -m compert.train --dataset_path datasets/GSM_new.h5ad       --save_dir /tmp --max_epochs 1  --doser_type sigm
python -m compert.train --dataset_path datasets/pachter_new.h5ad       --save_dir /tmp --max_epochs 1  --doser_type logsigm
python -m compert.train --dataset_path datasets/cross_species_new.h5ad       --save_dir /tmp --max_epochs 1  --doser_type mlp --split_key split4

python -m compert.train --dataset_path datasets/sciplex3_new.h5ad       --save_dir /tmp --max_epochs 1  --doser_type sigm
python -m compert.train --dataset_path datasets/sciplex3_new.h5ad       --save_dir /tmp --max_epochs 1  --doser_type sigm --split_key split1

python -m compert.train --dataset_path datasets/Norman2019_prep_new.h5ad       --save_dir /tmp --max_epochs 1  --doser_type linear

