#!/bin/bash

# rm -rf /checkpoint/$USER/sweep_GSM_2k_hvg
# rm -rf /checkpoint/$USER/sweep_GSM_4k_hvg
# rm -rf /checkpoint/$USER/sweep_pachter
# rm -rf /checkpoint/$USER/sweep_cross_species
# rm -rf /checkpoint/$USER/sweep_Norman2019
# rm -rf /checkpoint/$USER/sweep_sciplex3_prepared
# rm -rf /checkpoint/$USER/sweep_sciplex3_prepared_logsigm




python -m cpa.sweep --data datasets/GSM_new.h5ad               --save_dir /checkpoint/$USER/sweep_GSM_new_logsigm        --doser_type logsigm   --max_minutes 120
python -m cpa.sweep --data datasets/pachter_new.h5ad           --save_dir /checkpoint/$USER/sweep_pachter_new_logsigm    --doser_type logsigm   --max_minutes 120
python -m cpa.sweep --data datasets/cross_species_new.h5ad     --save_dir /checkpoint/$USER/sweep_cross_species_new      --doser_type mlp       --max_minutes 120
for i in {1..4}
do
   python -m cpa.sweep --data datasets/cross_species_new.h5ad     --save_dir /checkpoint/$USER/sweep_cross_species_new_split$i --doser_type mlp --split_key split$i --max_minutes 120
done


python -m cpa.sweep --data datasets/Norman2019_prep_new.h5ad        --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new_relu  --decoder_activation ReLU  --doser_type linear --max_minutes 300

for i in 1 21 22
do
   python -m cpa.sweep --data datasets/Norman2019_prep_new.h5ad     --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new_relu_split$i --doser_type linear --split_key split$i  --decoder_activation ReLU --max_minutes 300
done

for i in 12 16 20 24 28 4 8
do
   python -m cpa.sweep --data datasets/sciplex3_old_reproduced.h5ad     --save_dir /checkpoint/$USER/sweep_sciplex3_old_reproduced_split$i --doser_type logsigm --split_key split$i  --max_minutes 300
done


for i in {1..28}
do
   python -m cpa.sweep --data datasets/sciplex3_old_reproduced.h5ad     --save_dir /checkpoint/$USER/sweep_sciplex3_old_reproduced_split$i --doser_type logsigm --split_key split$i  --max_minutes 300
done


for i in {2..20}
do
  python -m cpa.sweep --data datasets/Norman2019_prep_new.h5ad     --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new_relu_split$i --doser_type linear --decoder_activation ReLU  --split_key split$i --max_minutes 300
done

python -m cpa.sweep --data datasets/Norman2019_prep_new.h5ad        --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new_relu_split23  --split_key 23 --decoder_activation ReLU  --doser_type linear --max_minutes 300




python -m cpa.sweep --data datasets/sciplex3_old_reproduced.h5ad       --save_dir /checkpoint/$USER/sweep_sciplex3_old_reproduced_logsigm          --doser_type logsigm   --max_minutes 3000
python -m cpa.sweep --data datasets/sciplex3_old_reproduced.h5ad     --save_dir /checkpoint/$USER/sweep_sciplex3_old_reproduced_sigm                 --doser_type sigm      --max_minutes 3000

python -m cpa.sweep --data datasets/pachter_new.h5ad           --save_dir /checkpoint/$USER/sweep_pachter_new            --doser_type sigm      --max_minutes 120
python -m cpa.sweep --data datasets/GSM_new.h5ad               --save_dir /checkpoint/$USER/sweep_GSM_new                --doser_type sigm      --max_minutes 120
python -m cpa.sweep --data datasets/sciplex3_new.h5ad       --save_dir /checkpoint/$USER/sweep_sciplex3_new                  --doser_type sigm      --max_minutes 300

python -m cpa.sweep --data datasets/lincs.h5ad     --save_dir /checkpoint/$USER/sweep_lincs_logsigm                 --doser_type logsigm      --max_minutes 3000
python -m cpa.sweep --data datasets/lincs.h5ad     --save_dir /checkpoint/$USER/sweep_lincs_sigm                 --doser_type sigm      --max_minutes 3000

