"""
Functions used by misalignment.py to compute and record the cosines of the angles between the movement of generated samples under
SDG/Adam updates of a WGAN generator and the directions of the critic's gradient. 
"""

import numpy as np
import matplotlib.pyplot as plt
import torch




def cosines_histogram(crit_vecs, sgd_vecs, adam_vecs, iteration, hist_save_path, args):
    """
    Computes cosines values and their means and standard deviations of the vectors specified by the first three inputs. Makes and 
    saves a histogram of the cosine values. 
    INPUTS
    (The first three inputs are torch tensors of shape [num_samples x data_shape])
        - crit_vecs: The values of the critic's gradient evaluated at generated samples.
        - sgd_vecs: Vectors in the directions of minus the updates of the generated samples if SGD was used to train the generator.
        - adam_vecs: The actual updates of the generated samples obtained using Adams
        - iteration: Current training iteration (to be recorded in the histogram).
        - hist_save_path: Path where the histogram will be saved.
        - args: Arguments to the main program (i.e. misalignments.py). Used in the histogram.
    OUTPUTS
        - SGD cosine values in a one dimensional numpy array.
        - Statistics of SGD cosine values (mean, std).
        - Adam cosine values in a one dimensional numpy array.
        - Statistics of Adam cosine values (mean, std).
    """
    # Normalize all vectors
    crit_vecs = normalize(crit_vecs)
    sgd_vecs = normalize(sgd_vecs)
    adam_vecs = normalize(adam_vecs)

    # Compute cosines
    bs = crit_vecs.size(0)
    
    sgd_cosines = torch.sum((crit_vecs * sgd_vecs).view(bs, -1), dim=1).cpu().data.numpy()
    sgd_ave = np.format_float_positional(np.average(sgd_cosines), 4)
    sgd_std = np.format_float_positional(np.std(sgd_cosines), 4)
    
    adam_cosines = torch.sum((crit_vecs * adam_vecs).view(bs, -1), dim=1).cpu().data.numpy()
    adam_ave = np.format_float_positional(np.average(adam_cosines), 4)
    adam_std = np.format_float_positional(np.std(adam_cosines), 4)
    
    # Make histogram
    text = 'Alignment cosines\nat iteration {}\nModel: {}\nDataset: {}'.format(iteration, args.model, args.target)
    plt.hist([sgd_cosines, adam_cosines], 40, range=(-1,1), label=['SGD','Adam'])
    plt.xlabel('Magnitude of normalized dot products')
    plt.legend(loc='upper left')
    plt.figtext(x=0.15,y=0.6, s=text, fontsize=9)
    
    # Save histogram
    plt.savefig(hist_save_path)
    plt.close()
    
    return sgd_cosines, (sgd_ave, sgd_std), adam_cosines, (adam_ave, adam_std)



def normalize(data):
    """
    INPUTS
        - data; data to be normalized. Assumed to be N distinct tensors of
          shape C*H*W, and each one is normalized by its Euclidean norm
    OUTPUTS
        - normalized_data; tensor of same shape as data, but torch.norm(*.view(N,-1), dim = 1) = 1
    """
    bs = data.size(0)
    data_norms = torch.norm(data.view(bs, -1), dim = 1)
    data_norms = data_norms.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    data_norms = data_norms.expand_as(data)
    return (data/data_norms).detach().clone()


