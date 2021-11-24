import torch
from torchvision import utils
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# For generating samples
def generate_image(frame, data, ext, path):
    """After re-centering data, this code saves a grid of samples to the path
    path/samples/"""

    data = 0.5*data + 0.5*torch.ones_like(data) # by default, data is generated in [-1,1]

    grid = utils.make_grid(data, nrow = 16, padding = 1)
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
    plt.axis('off')
    os.makedirs(os.path.join(path,'samples'), exist_ok = True)
    plt.savefig(os.path.join(path,'samples/{}.{}'.format(frame,ext)), dpi = 300)
    plt.close()


def generate_image_v2(im_name, data, ext, path):
    """
    After re-centering data, this code saves a grid of samples to the specified path.
    Saves grid under path/im_name.ext.
    """

    data = 0.5*data + 0.5*torch.ones_like(data) # by default, data is generated in [-1,1]

    grid = utils.make_grid(data, nrow = 16, padding = 1)
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.savefig(os.path.join(path,'{}.{}'.format(im_name,ext)), dpi = 300)
    plt.close()




def save_individuals(b_idx, t_idx, data, ext, path, to_rgb = False):
    """After re-centering data, this code individually saves numbered images in a path
    of the form path/timestamp{}/. Optionally will repeat greyscale images 3 times along colour channel"""
    data = 0.5*data + 0.5*torch.ones_like(data) # by default, data is generated in [-1,1]
    bs = data.shape[0]
    os.makedirs(os.path.join(path, 'pics'), exist_ok = True)
    path = os.path.join(path, 'pics/timestamp{}'.format(t_idx))
    os.makedirs(path, exist_ok = True)

        
    for i in range(bs):
        single = data[i,:,:,:]
        if single.shape[0] == 1 and to_rgb:
            single = single.repeat(3,1,1)

        utils.save_image(single, os.path.join(path, '{:05d}.{}'.format(b_idx*bs + i, ext)))



def save_individuals_v2(b_idx, data, ext, path, to_rgb=False):
    """After re-centering data, this code individually saves numbered images in the specified path.
       Optionally will repeat greyscale images 3 times along colour channel"""
    data = 0.5*data + 0.5*torch.ones_like(data) # by default, data is generated in [-1,1]
    bs = data.shape[0]
   
    for i in range(bs):
        single = data[i,:,:,:]
        if single.shape[0] == 1 and to_rgb:
            single = single.repeat(3,1,1)
        utils.save_image(single, os.path.join(path, '{:05d}.{}'.format(b_idx*bs + i, ext)))




    
