#!/bin/bash

python -m cpa.collect_results     --save_dir /checkpoint/$USER/sweep_GSM_new_logsigm
python -m cpa.collect_results     --save_dir /checkpoint/$USER/sweep_pachter_new
python -m cpa.collect_results     --save_dir /checkpoint/$USER/sweep_cross_species_new
for i in {1..4}
do
   python -m cpa.collect_results  --save_dir /checkpoint/$USER/sweep_cross_species_new_split$i 
done

python -m cpa.collect_results     --save_dir /checkpoint/$USER/sweep_sciplex3_new_logsigm
python -m cpa.collect_results     --save_dir /checkpoint/$USER/sweep_sciplex3_new

python -m cpa.collect_results     --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new
python -m cpa.collect_results     --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new_relu

python -m cpa.collect_results     --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new_split1

for i in 1 21 22
do
   python -m cpa.collect_results   --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new_relu_split$i
done

for i in {1..28}
do
   python -m cpa.collect_results  --save_dir /checkpoint/$USER/sweep_sciplex3_old_reproduced_split$i
done

for i in {2..23}
do
   python -m cpa.collect_results  --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new_split$i
done



for i in {2..20}
do
   python -m cpa.collect_results  --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new_relu_split$i
done

python -m cpa.collect_results     --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new_relu_split23


python -m cpa.collect_results     --save_dir /checkpoint/$USER/sweep_sciplex3_new
python -m cpa.collect_results     --save_dir /checkpoint/$USER/sweep_GSM_new
python -m cpa.collect_results     --save_dir /checkpoint/$USER/sweep_pachter_new_logsigm


python -m cpa.collect_results     --save_dir /checkpoint/$USER/sweep_sciplex3_old_reproduced_logsigm
python -m cpa.collect_results     --save_dir /checkpoint/$USER/sweep_sciplex3_old_reproduced_sigm

python -m cpa.collect_results     --save_dir /checkpoint/$USER/sweep_lincs_logsigm
python -m cpa.collect_results     --save_dir /checkpoint/$USER/sweep_lincs_sigm

python -m cpa.collect_results     --save_dir /checkpoint/$USER/kang_split




