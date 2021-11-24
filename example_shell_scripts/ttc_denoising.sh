##### This file is not intended to be used to run an experiment. It provides an example on how to train TTC critics for image denoise with ttc.py and how to evaluate the denoise performance using denosing_eval.py #####

#!/bin/bash

## Prepare virtual environment
source #path/to/virtualenv/bin/activate

#Variables
target='bsds500'
sources='noisybsds500'
temp_dir='some/path'
sigma=0.3
lamb=1000.
theta=0.7
seed=0
critters=5000
num_crit=20
model='arConvNet'
bs=16

results_folder="some/other/path"
target_folder="path/to/data"

mkdir $results_folder   ## If it doesn't arleady exist
mkdir "$results_folder/not_model_dicts"

#Prepare data
cp -r  $target_folder $temp_dir
unzip -q  "$tempdir/${target}.zip" -d "$temp_dir"

#Run code
python ttc.py --target $target --source $sources --temp_dir $temp_dir --data $temp_dur --lamb $lamb --critters $critters --plus --num_crit $num_crit --theta $theta --seed $seed --bs $bs --sigma $sigma --model $model

cp -r $temp_dir/model_dicts "$results_folder"
cp -r $temp_dir/log.pkl "$results_folder/not_model_dicts/"
cp -r $temp_dir/train_config.txt "$results_folder/not_model_dicts/"

#evaluate metrics
python denoise_eval.py --temp_dir $temp_dir --data $temp_dir --model $model --bs 200  --seed $seed --sigma $sigma

cp -r $temp_dir/samples "$results_folder/not_model_dicts/"
cp -r $temp_dir/metrics.pkl "$results_folder/not_model_dicts/"
