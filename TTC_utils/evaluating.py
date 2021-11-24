"""
FID evaluation function used by misalignments.py
"""

import os
import shutil
import numpy as np
import torch
from pytorch_fid import fid_score
from generate_samples import save_individuals_v2


def fid_eval(generator,
             real_data,
             directory,
             inception_net,
             noise):
    """
    Will compute FID of generated distribution from generator when compared to real_data.
    INPUTS
        - generator: A generator network to be evaluated.
        - real_data: Path either of a folder with the test data against which the generated samples are to be compared, or
                     a .npz file containing the inception feature-space statistics of the test data (avoinding computing
                     those again).
        - directory: Path of a folder where the generated samples will be saved (in a subfolder), as well as the inception
                     feature-space statistics of the test data.
        - inception_net: An instance of the InceptionV3 network.
        - noise: Generator noise inputs used to generate samples to be compared with the test data.
    OUTPUTS
        - fid: The computed FID
        - real_data: The new path for test data, which will have changed if the real_data input specifed a folder containing
                     the test data.
    """
    # ------------------------------- Setting up -------------------------------
    # Choose device to use when evaluating FIDs
    use_cuda = next(generator.parameters()).is_cuda
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    
    # ------------------------------- Generating fake data -------------------------------
    # Make folder in which generated samples will be saved, deleting previous one if necessary
    fake_samples_path = os.path.join(directory, 'samples_for_fid')
    if os.path.exists(fake_samples_path):
        shutil.rmtree(os.path.join(fake_samples_path))
    os.makedirs(fake_samples_path)
    
    fake = generator(noise)
    save_individuals_v2(0, 
                        data=fake, 
                        ext='jpg', 
                        path=fake_samples_path, 
                        to_rgb=True if fake.shape[1]==1 else False)
    
    # ------------------------------- Evaluating FID -------------------------------
    real_mean, real_cov = fid_score.compute_statistics_of_path(real_data, inception_net, 100, 2048, device)
    if not real_data.split('.')[-1] == 'npz':
        real_data = os.path.join(directory, 'real_data_stats.npz')
        np.savez(real_data, mu=real_mean, sigma=real_cov)    
    fake_mean, fake_cov = fid_score.compute_statistics_of_path(fake_samples_path, inception_net, 100, 2048, device)
    fid = fid_score.calculate_frechet_distance(fake_mean, fake_cov, real_mean, real_cov)
    return fid, real_data