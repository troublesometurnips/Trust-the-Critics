##### This file is not intended to be used to run an experiment. It only provides an approximative example of how to train TTC critics for image generation with ttc.py and how to evaluate the generative performance using ttc_eval.py #####

#!/bin/bash

## Prepare virtual environment
source #path/to/virtualenv/bin/activate

#Variables
target='mnist'
source='noise'
temp_dir='some/path'
lamb=1000.
theta=0.9
seed=1
critters=5000
num_crit=20
model='infogan'
bs=50
eval_freq=2

timestamp=`date '+%Y%m%d%H%M%S'`
results_folder="some/path/where/we/copy/results"
target_folder="some/path/to/dataset"

mkdir $results_folder
mkdir "$results_folder/not_model_dicts"

#Prepare data
cp -r  $target_folder "$temp_dir/MNIST"
unzip -q  "$temp_dir/MNIST/mnisttest.zip" -d "$temp_dir/mnisttest" ## Data for FID evaluation also contained in $target_folder

#Run code
python ttc.py --target $target --source $source --temp_dir $temp_dir --data $temp_dir --lamb $lamb --critters $critters --plus --num_crit $num_crit --theta $theta --seed $seed --bs $bs --model $model

cp -r $temp_dir/model_dicts "$results_folder"
cp -r $temp_dir/log.pkl "$results_folder/not_model_dicts/"
cp -r $temp_dir/train_config.txt "$results_folder/not_model_dicts/"

#evaluate metrics
python ttc_eval.py --target $target --source $source --temp_dir $temp_dir --data $temp_dir --model $model --bs 1000 --FID --numsample 10000 --eval_freq $eval_freq --seed $seed

cp -r $temp_dir/samples "$results_folder/not_model_dicts/"
cp -r $temp_dir/metrics.pkl "$results_folder/not_model_dicts/"
