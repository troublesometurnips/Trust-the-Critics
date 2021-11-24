##### This file is not intended to be used to run an experiment. It provides an example on how to train TTC critics for image denoise with ttc.py and how to evaluate the denoise performance using denosing_eval.py #####

#!/bin/bash

## Prepare virtual environment
source #path/to/virtualenv/bin/activate

#Variables
target='monet'
source='photo'
temp_dir='some/dir'
lamb=1000.
theta=0.9
seed=-1
critters=5000
num_crit=30
model='sndcgan'
bs=16
eval_freq=1

timestamp=`date '+%Y%m%d%H%M%S'`
results_folder="./results/$target/$timestamp/"
target_folder="some/path/photo2monet/monet.zip"
source_folder="some/path/photo2monet/photo.zip"

mkdir $results_folder # If it doesn't already exist
mkdir "$results_folder/not_model_dicts"

cp -r  $target_folder $temp_dir/
unzip -q "$temp_dir/monet.zip" -d $tempdir
cp -r  $source_folder $temp_dir
unzip -q "$temp_dir/photo.zip" -d $temp_dir

#Run code
python ttc.py --target $target --source $source --temp_dir $temp_dir --data $temp_dir --lamb $lamb --critters $critters --plus --num_crit $num_crit --theta $theta --seed $seed --bs $bs --model $model

cp -r $temp_dir/model_dicts "$results_folder"
cp -r $temp_dir/log.pkl "$results_folder/not_model_dicts/"
cp -r $temp_dir/train_config.txt "$results_folder/not_model_dicts/"

#evaluate metrics
python ttc_eval.py --target $target --source $source --temp_dir $temp_dir --data $temp_dir --model $model --bs 32 --eval_freq $eval_freq --seed $seed

cp -r $temp_dir/samples "$results_folder/not_model_dicts/"

python experiment_tracker.py --results_folder $results_folder --df_path $logfile --experiment_name $exp_name
