#!/bin/bash

# change the path vairable to your path to the datasets folder
path='../cpa_binaries'

python -m compert.train --dataset_path $path/datasets/GSM_new.h5ad       --save_dir /tmp --max_epochs 100  --doser_type sigm
python -m compert.train --dataset_path $path/datasets/pachter_new.h5ad       --save_dir /tmp --max_epochs 1  --doser_type logsigm
python -m compert.train --dataset_path $path/datasets/cross_species_new.h5ad       --save_dir /tmp --max_epochs 1  --doser_type mlp --split_key split4

python -m compert.train --dataset_path $path/datasets/sciplex3_new.h5ad       --save_dir /tmp --max_epochs 1  --doser_type sigm
python -m compert.train --dataset_path $path/datasets/sciplex3_new.h5ad       --save_dir /tmp --max_epochs 1  --doser_type sigm --split_key split1

python -m compert.train --dataset_path $path/datasets/Norman2019_prep_new.h5ad       --save_dir /tmp --max_epochs 1  --doser_type linear

