##### This file is not intended to be used to run an experiment. It only provides an approximative example of how to run the misalignments.py script #####

#!/bin/bash

## Prepare virtual environment
source #path/to/virtualenv/bin/activate

## Select hyperparameters and other inputs
seed=12345
target='mnist'
model='infogan'
temp_dir='some/path/maybe/on/a/compute/node'
results_path='some/local/path'
checkpoints='1_10_50_200'
critters=5
start_critters=0
lamb=10.
bs=128

## Prepare data
target_folder="path/to/mnist/dataset"
cp -r $target_folder "$temp_dir/MNIST"
unzip -q "some/path/to/testset" -d "$temp_dir/mnisttest"
fid_data="$temp_dir/mnisttest"

## Run program
python misalignments.py --seed $seed --target $target --data $temp_dir --fid_data $fid_data --temp_dir $temp_dir --results_path $results_path --model $model --checkpoints $checkpoints --critters $critters --start_critters $start_critters --lamb $lamb --bs $bs
