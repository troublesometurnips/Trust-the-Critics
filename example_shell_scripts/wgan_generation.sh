##### This file is not intended to be used to run an experiment. It only provides an approximative example of how to train a WGAN for image generation with wgan_gp.py and how to evaluate the generative performance using wgan_gp_eval.py #####

#!/bin/bash

## Prepare virtual environment
source #path/to/virtualenv/bin/activate

#Variables
target='mnist'
temp_dir='some/path'
model='infogan'
lamb=1000.
seed=1
critters=5
bs=50
geniters=1000
glr=0.001
eval_freq=1

results_folder="some/other/path"
target_folder="path/to/data"

mkdir $results_folder # If it doesn't already exist.
mkdir "$results_folder/not_model_dicts"

#Prepare data
cp -r  $target_folder "$temp_dir/MNIST"
unzip -q  "$temp_dir/MNIST/mnisttest.zip" -d "$temp_dir/mnisttest" # Here we assume the test data for FID evaluation is archived in MNIST/mnisttest.zip

#Run code
python wgan_gp.py --target $target --temp_dir $temp_dir --data $temp_dir --lamb $lamb --critters $critters --plus --seed $seed --bs $bs --iters $geniters --glr $glr

cp -r $temp_dir/model_dicts "$results_folder"
cp -r $temp_dir/log.pkl "$results_folder/not_model_dicts/"
cp -r $temp_dir/train_config.txt "$results_folder/not_model_dicts/"

#evaluate metrics
python wgan_gp_eval.py --target $target  --temp_dir $temp_dir --data $temp_dir --bs 1000 --FID --bigsample --eval_freq $eval_freq --seed $seed --model $model

cp -r $temp_dir/samples "$results_folder/not_model_dicts/"
cp -r $temp_dir/metrics.pkl "$results_folder/not_model_dicts/"
