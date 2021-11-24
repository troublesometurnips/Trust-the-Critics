"""
This script trains a WGAN and computes the misalignments between the movement of generated samples caused by updating the generator
(with both SGD and Adam) and the optimal directions given by the gradient of the critic. The misalignments are computed in the 
form of the cosines of the angles between the directions of SGD/Adam updates of generated samples and the directions of the critic's
gradient. The generator's performance is evaluated throughout training using FID. Note that SGD generator updates are only computed
to get the misalignments, while Adam generator updates are acutally used to update the generator's parameters.

IMPORTANT INPUTS
    - target (required): Dataset used for experiment. Can be either MNIST or Fashion-MNIST.
    - model (defaults to infogan): WGAN model used during experiments. Can be either infogan or dcgan.
    - seed (defaults to -1): Random seed that can be used for reproducibility. 
    - data (required): Directory where the dataset is located.
    - fid_data (required): Directory where the test data is evaluated (for FID evaluation).
    - temp_dir (defaults to current working directory): Directory used during computations, e.g. a temporary directory on a compute node.  If unspecified
                                   this defaults to the current working directory.
    - results_path (defaults to current working directory): Directory where the results of the experiment will be saved.
    - checkpoints (required): A string containing integers specifying the iterations at which misalignment cosines and FID will be 
                              computed. These integers should be separated by underscores. This input determines the runtime, since 
                              training will continue for as many iterations as specified by the largest integers. Furthermore, FID 
                              evaluation being computationally demanding, adding more checkpoints will lead to larger runtime, even
                              if the largest checkpoint remains constant.
    - bs (defaults to 128): Mini-batch size used to train both the generator and the critic. 
    - num_samples (defaults to 256): Number of generated samples at which misalignments are computed at each checkpoint. Note that
                                     these generated samples are drawn separately from those used to train the generator. 

OUTPUTS
Running this script will create a subdirectory of results_path with a name describing the target dataset and the WGAN model used.
The following files will be saved in this subdirectory:
    - A .pkl file with a pandas DataFrame containing the results obtained at each checkpoint. The DataFrame has one row named 
      after each checkpoint, and five columns. The first columns contains all computed cosine values for SGD. The second contains 
      the statistics of the cosine values for SGD in the form (mean, std). The next two columns are similar, but for Adam. The 
      last column contains the FIDs.
    - A .zip archive containing histograms of the cosine values computed at each checkpoint.
    - A .txt file containing the settings of the experiments and summarizing the results obtained at each checkpoint by giving
      the mean and standard deviation of cosine values for SGD and Adam, as well as the FID.
Running this file will also save the histograms in temp_dir.
"""

import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'TTC_utils'))
import shutil
import argparse
import time
import random
import datetime
import numpy as np
import pandas as pd
import torch
from torch import optim
from pytorch_fid import inception

import training
from evaluating import fid_eval
import dataloader
import compare_directions
import networks


starttime = time.time()
# ------------------------------- Command line arguments ------------------------------------------------
parser = argparse.ArgumentParser('WGAN training with misalignment computations')
parser.add_argument('--seed', type=int, default=-1, help='Set random seed for reproducability')
parser.add_argument('--target', type=str, required=True, choices=['mnist', 'fashion'])
parser.add_argument('--data', type=str, required=True, help='Directory where data is located')
parser.add_argument('--fid_data', type=str, required=True, help='Directory where test data for FID evaluation is located')
parser.add_argument('--temp_dir', type=str, default=None, help='Temporary directory for computations')
parser.add_argument('--results_path', type=str, default=None, help='Path where results will be saved')
parser.add_argument('--model', type=str, default='infogan', choices=['dcgan', 'infogan'])
parser.add_argument('--checkpoints', type=str, required=True, help='A string of the form i1_i2_i3_... where the ijs are integers specifying iterations')
parser.add_argument('--critters', type=int, default=5, help='number of critic iters per gen update')
parser.add_argument('--start_critters', type=int, default=0, help='number of critic iteration at the start')
parser.add_argument('--lamb', type=float, default=10., help='parameter multiplying gradient penalty')
parser.add_argument('--bs', type=int, default=128, help='batch size used to train')
parser.add_argument('--num_samples', type=int, default=256, help='number of samples where we compute updates')
parser.add_argument('--DIM', type=int, default=64, help='int determining network dimensions')
parser.add_argument('--num_workers', type=int, default=0, help='Number of data loader processes')
parser.add_argument('--beta_1_d', type=float, default=0.5, help='Parameter for Adam optimizer of the critic')
parser.add_argument('--beta_2_d', type=float, default=0.999, help='Parameter for Adam optimizer of the critic')
parser.add_argument('--beta_1_g', type=float, default=0.5, help='Parameter for Adam optimizer of the generator')
parser.add_argument('--beta_2_g', type=float, default=0.999, help='Parameter for Adam optimizer of the generator')
args = parser.parse_args()

# We parallelize if possible
use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')

# These are the training iterations where we compute misalignments (and FID if required)
checkpoints = args.checkpoints.split('_')
for i in range(len(checkpoints)):
    checkpoints[i] = int(checkpoints[i])
checkpoints.sort()

print('Computing misalignment of generator updates in WGAN training \n')
print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('Checkpoints: {}'.format(checkpoints))
print('use_cuda = {}'.format(use_cuda), '\n')


# ------------------------------- Setting up -------------------------------
# Temporary folder for computations. Might be on a computation node.
if args.temp_dir is None:
    temp_dir = os.getcwd()
else:
    temp_dir = args.temp_dir

# Everything will be saved in a sub-folder of:
if args.results_path is None:
    results_path = os.getcwd()
else:
    results_path = args.results_path

# Define experiment's name. If there's already an experiment with this name in results_path, add timestamp to the name.
if args.seed==-1:
    exp_name = 'misalignments_{}_{}'.format(args.model, args.target)
else:
    exp_name = 'misalignments_{}_{}_seed{}'.format(args.model, args.target, args.seed)

if exp_name in os.listdir(results_path):
    time_now = datetime.datetime.now()
    exp_name = exp_name + '_' + time_now.strftime("%Y%d%H%M%S")

# Make folder where we'll save everything.
save_path = os.path.join(results_path, exp_name)
os.makedirs(save_path)

# Make subfolder for histograms
hist_save_dir = os.path.join(temp_dir, 'histograms')
os.makedirs(hist_save_dir)

# For FID evaluations
real_data = args.fid_data
block_idx = inception.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
inception_net = inception.InceptionV3([block_idx]).to(device)


# ------------------------------- Create txt file and save settings in it -------------------------------
exp_file = open(os.path.join(save_path, '{}.txt'.format(exp_name)), 'w')
exp_file.write('Misalignments computation \n\n')
exp_file.write('Experiment name: ' + exp_name)
exp_file.write('\nSettings: \n')
for p in vars(args).items():
    exp_file.write("  {}: {}\n".format(p[0], p[1]))
exp_file.write('Checkpoints: \n{}\n\n'.format(checkpoints))
exp_file.close()


# ------------------------------- Deterministic behavior -------------------------------
if args.seed != -1: #if non-default seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False # If true, optimizes convolution for hardware, but gives non-deterministic behaviour
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


# ------------------------------- Defining dataloader, networks, optimizers ------------------------------- 
# Dataloader
loader = getattr(dataloader, args.target)(args, train=True)
num_chan = loader.in_channels
hpix = loader.hpix
wpix = loader.wpix
loader_gen = iter(loader)

# Networks
generator = getattr(networks, args.model + '_generator')(args.DIM, num_chan, hpix, wpix)
critic = getattr(networks, args.model)(args.DIM, num_chan, hpix, wpix)
latent_dim = generator.latent_dim

if use_cuda:
    generator = generator.cuda()
    critic = critic.cuda()
networks = (generator, critic)

# Optimizers
optim_g = optim.Adam(generator.parameters(), lr=1e-4, betas=(args.beta_1_d, args.beta_2_d))
optim_d = optim.Adam(critic.parameters(), lr=1e-4, betas=(args.beta_1_g, args.beta_2_g))
optimizers = (optim_g, optim_d)

# Noise inputs used for FID evaluations
fid_noise = torch.randn(10000, latent_dim)
if use_cuda:
    fid_noise = fid_noise.cuda()


# ------------------------------- Training and computing alignments ------------------------------- 
# Start iterations if requested
if args.start_critters > 0:
    loader_gen = training.train_iters(networks=networks,
                                      optimizers=optimizers,
                                      latent_dim=latent_dim,
                                      loader=loader,
                                      loader_gen=loader_gen,
                                      iters=1,
                                      critters=args.start_critters,
                                      lamb=args.lamb,
                                      skip_last=True)

current_iter = 0
df = pd.DataFrame(columns = ['SGD_cosines', 'SGD_cosines_stats', 'Adam_cosines', 'Adam_cosines_stats', 'FID'])
for i in range(len(checkpoints)):
    num_iters = checkpoints[i]-current_iter 
    
    # Train until next checkpoint
    loader_gen = training.train_iters(networks=networks,
                                      optimizers=optimizers,
                                      latent_dim=latent_dim,
                                      loader=loader,
                                      loader_gen=loader_gen,
                                      iters=num_iters,
                                      critters=args.critters,
                                      lamb=args.lamb,
                                      skip_last=True) # skip_last generator update because we use misalignment_update instead.
    current_iter += num_iters
    
    # ------------------------------- Checkpoints ------------------------------- 
    print("Computing misalignments during generator update {}:".format(current_iter))
    
    # Series to record results, will be appended to df
    series = pd.Series(name=current_iter)
    
    # Get three types of updates
    noise1 = torch.randn(args.num_samples, latent_dim)
    noise2 = torch.randn(args.bs, latent_dim)
    if use_cuda:
        noise1 = noise1.cuda()
        noise2 = noise2.cuda()
    crit_grads, sgd_updates, adam_updates = training.misalignment_iter(critic, generator, optim_g, noise1, noise2)
    
    # Make histograms and get statistics
    hist_save_path = os.path.join(hist_save_dir, 'cosines_at_iter_{}'.format(current_iter))
    sgd_cosines, sgd_cosines_stats, adam_cosines, adam_cosines_stats = compare_directions.cosines_histogram(crit_grads, sgd_updates, adam_updates, current_iter, hist_save_path, args)
    
    # Record results
    series['SGD_cosines'], series['SGD_cosines_stats'] = sgd_cosines, sgd_cosines_stats
    series['Adam_cosines'], series['Adam_cosines_stats'] = adam_cosines, adam_cosines_stats
    print('SGD alignment cosines stats (mean, std) = {} \nAdam alignment cosines stats (mean, std) = {}'.format(sgd_cosines_stats, adam_cosines_stats))
    
    # Evaluate FID
    print("Evaluating FID")
    fid, real_data = fid_eval(generator=generator,
                              real_data=real_data,
                              directory=temp_dir,
                              inception_net=inception_net,
                              noise=fid_noise)
    series['FID'] = fid
    print("FID obtained: {}".format(fid))
    
    df = df.append([series])
    
    # Write results in txt file
    exp_file = open(os.path.join(save_path, '{}.txt'.format(exp_name)), 'a')
    exp_file.write("Generator update {}\n".format(current_iter))
    exp_file.write("SGD alignment cosines stats (mean, std) = {} \nAdam alignment cosines stats (mean, std) = {}\nFID after update = {}\n\n".format(sgd_cosines_stats, adam_cosines_stats, fid))
    exp_file.close()
    
    

# ------------------------------- Save results in save_path -------------------------------
df.to_pickle(os.path.join(save_path, '{}_frame.pkl'.format(exp_name)))
shutil.make_archive(os.path.join(save_path, '{}_histograms'.format(exp_name)), 
                    'zip', os.path.join(args.temp_dir, 'histograms'))

# Record time needed
total_time = time.time() - starttime
exp_file = open(os.path.join(save_path, '{}.txt'.format(exp_name)), 'a')
exp_file.write('Total time required: {}s'.format(total_time))
exp_file.close()
print("\n\nTotal time needed: {}s".format(total_time))
    




    
    
    
    

    
    
   
    
    
    
    






