# Trust-the-Critics
This repository is a PyTorch implementation of the TTC algorithm and the WGAN misalignment experiments presented in *Trust the Critics: Generatorless and Multipurpose WGANs with Initial Convergence Guarantees*.


## How to run this code ##
* Create a Python virtual environment with Python 3.8 installed.
* Install the necessary Python packages listed in the requirements.txt file (this can be done through pip install -r /path/to/requirements.txt).

In the example_shell_scripts folder, we include samples of shell scripts we used to run our experiments. We note that training  generative models is computationally demanding, and thus requires adequate computational resources (i.e. running this on your laptop is not recommended).


### TTC algorithm
The various experiments we run with TTC are described in Section 5 and Addendix B of the paper. Illustrating the flexibility of the TTC algorithm, the image generation, denoising and translation experiments can all be run using the ttc.py script; the only necessary changes are the source and target datasets. Running TTC with a given source and a given target will train and save several critic neural networks that can subsequently be used to push the source distribution towards the target distribution by applying the 'steptaker' function found in TTC_utils/steptaker.py once for each critic.  

Necessary arguments for ttc.py are:
* 'source' : The name of the distribution or dataset that is to be pushed towards the target (options are listed in ttc.py).
* 'target' : The name of the target dataset (options are listed in ttc.py).
* 'data' : The path of a directory where the necessary data is located. This includes the target dataset, in a format that can be accessed by a dataloader object obtained from the corresponding function in dataloader.py. Such a dataloader always belongs to the torch.utils.data.DataLoader class (e.g. if target=='mnist', then the corresponding dataloader will be an instance of torchvision.datasets.MNIST, and the MNIST dataset should be placed in 'data' in a way that reflects this). If the source is a dataset, it needs to be placed in 'data' as well. If source=='untrained_gen', then the untrained generator used to create the source distribution needs to be saved under 'data/ugen.pth'.
* 'temp_dir' : The path of a directory where the trained critics will be saved, along with a few other files (including the log.pkl file that contains the step sizes). Despite the name, this folder isn't necessarily temporary.

Other optional arguments are described in a commented section at the top of the ttc.py script. Note that running ttc.py will only train the critics that the TTC algorithm uses to push the source distribution towards the target distribution, it will not actually push any samples from the source towards the target (as mentioned above, this is done using the steptaker function).


**TTC image generation**   
For a generative experiment, run ttc.py with the source argument set to either 'noise' or 'untrained_gen' and the target of your choice. Then, run ttc_eval.py, which will use the saved critics and step sizes to push noise inputs towards the target distribution according to the TTC algorithm (using the steptaker function), and which will optionally evaluate generative performance with FID and/or MMD (FID is used in the paper). The arguments 'source', 'target', 'data', 'temp_dir' and 'model' for ttc_eval.py should be set to the same values as when running ttc.py. If evaluating FID, the folder specified by 'temp_dir' should contain a subdirectory named 'temp_dir/{target}test' (e.g. 'temp_dir/mnisttest' if target=='mnist') containing the test data from the target dataset saved as individual files. For instance, this folder could contain files of the form '00001.jpg', '00002.jpg', etc. (although extensions other than .jpg can be used).



**TTC denoising**  
For a denoising experiment, run ttc.py with source=='noisybsds500' and target=='bsds500' (specifying a noise level with the 'sigma' argument). Then, run denoise_eval.py (with the same 'temp_dir', 'data' and 'model' arguments), which will add noise to images, denoise them using the TTC algorithm and the saved critics, and evaluate PSNR's. 



**TTC Monet translation**  
For a denoising experiment, run ttc.py with source=='photo' and target=='monet'. Then run ttc_eval.py (with the same 'source', 'target', 'temp_dir', 'data' and 'model' arguments, and presumably with no FID or MMD evaluation), which will sample realistic images from the source and make them look like Monet paintings.



### WGAN misalignment 
The WGAN misalignment experiments are described in Section 3 and Appendix B.1 of the paper, and are run using misalignments.py. This script trains a WGAN while, at some iterations, measuring how misaligned the movement of generated samples caused by updating the generator is from the critic's gradient. The generator's FID is also measured at the same iterations.

The required arguments for misalignments.py are:
* 'target' : The dataset used to train the WGAN - can be either 'mnist' or 'fashion' (for Fashion-MNIST).
* 'data' : The path of a folder where the MNIST (or Fashion-MNIST) dataset is located, in a format that can be accessed by an instance of the torchvision.datasets.MNIST class (resp torchvision.datasets.FashionMNIST). 
* 'fid_data' :  The path of a folder containing the test data from the MNIST dataset saved as individual files. For instance, this folder could contain files of the form '00001.jpg', '00002.jpg', etc. (although extensions other than .jpg can be used).
* 'checkpoints' : A string of integers separated by underscores. The integers specify the iterations at which misalignments and FID are computed, and training will continue until the largest iteration is reached.

Other optional arguments (including 'results_path' and 'temp_dir') are described in a commented section at the top of the misalignments.py. The misalignment results reported in the paper (Tables 1 and 5, and Figure 3), correspond to using the default hyperparameters and  to setting the 'checkpoints' argument roughly equal to '10_25000_40000', with '10' corresponding the early stage in training, '25000' to the mid stage, and '40000' to the late stage. 



### WGAN generation 
For completeness we include the code that was used to obtain the WGAN FID statistics in Table 3 of the paper, which includes the wgan_gp.py and wgan_gp_eval.py scripts. The former trains a WGAN with the InfoGAN architecture on the dataset specified by the 'target' argument, saving generator model dictionaries in the folder specified by 'temp_dir' at ten equally spaced stages in training. The wgan_gp_eval.py script evaluates the performance of the generator with the different model dictionaries in 'temp_dir'. 

The necessary arguments to run wgan_gp.py are:
* 'target' : The name of the dataset to generate (can be either 'mnist', 'fashion' or 'cifar10').
* 'data' : Folder where the dataset is located.
* 'temp_dir' : Folder where the model dictionaries are saved.

Once wgan_gp.py has run, wgan_gp_eval.py should be called with the same arguments for 'target', 'data' and 'temp_dir', and setting the 'model' argument to 'infogan'. If evaluating FID, the 'temp_dir' folder needs to contain the test data from the target dataset saved as individual files. For instance, this folder could contain files of the form '00001.jpg', '00002.jpg', etc. (although extensions other than .jpg can be used).
  
  

## Reproducibility
This repository contains two branches: 'main' and 'reproducible'. You are currectly viewing the 'main' branch, which contains a clean version of the code meant to be easy to read and interpret and to run more efficiently than the version on the 'reproducible' branch. The results obtained by running the code on this branch are nearly (but not perfectly) identical to the results stated in the papers, the differences stemming from the randomness inherent to the experiments. The 'reproducible' branch (available at https://anonymous.4open.science/r/Trust-the-Critics-354B) allows one to replicate exactly the results stated in the paper (random seeds are specified) for the TTC experiments. 



## Computing architecture and running times
We ran different versions of the code presented here on Compute Canada (https://www.computecanada.ca/) clusters, always using a single NVIDIA V100 Volta or NVIDIA A100 Ampere GPU. Here are rough estimations of the running times for our experiments.

- **MNIST/Fashion MNIST generation training run (TTC)**: 60-90 minutes.
- **MNIST/Fashion MNIST generation training run (WGAN)**: 45-90 minutes (this includes misalignments computations).
- **CIFAR10 generation training run**: 3-4 hours (TTC), 90 minutes (WGAN-GP).
- **Image translation training run**: up to 20 hours.
- **Image denoising training run**: 8-10 hours.



## Assets 
Portions of this code, as well as the datasets used to produce our experimental results, make use of existing assets. We provide here a list of all assets used, along with the licenses under which they are distributed, if specified by the originator:
- The code used for training a WGAN as a baseline was initially built from a PyTorch implementation (https://github.com/caogang/wgan-gp) of WGAN-GP ((c) 2017 Ishaan Gulrajani). Distributed under the MIT licence
- **mmd.py**: from https://github.com/EmoryMLIP/OT-Flow, ((c) 2020 EmoryMLIP). Distributed under the MIT licence. Unused in the paper, but provides a separate interesting metric for measuring performance.
- **pytorch_fid**: from https://github.com/mseitzer/pytorch-fid. Distributed under the Apache License 2.0.
- **MNIST dataset**: from http://yann.lecun.com/exdb/mnist/. Distributed under the Creative Commons Attribution-Share Alike 3.0 license.
- **Fashion MNIST datset**: from  https://github.com/zalandoresearch/fashion-mnist ((c) 2017 Zalando SE, https://tech.zalando.com). Distributed under the MIT licence.
- **CIFAR10 dataset**: from https://www.cs.toronto.edu/~kriz/cifar.html.
- **Image translation datasets**: from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix ((c) 2017, Jun-Yan Zhu and Taesung Park). Distributed under the BSD licence.
- **BSDS500 dataset**: from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html.




