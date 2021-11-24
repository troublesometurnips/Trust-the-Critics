"""
WGAN training functions used by misalignments.py
"""

import os, sys
sys.path.append(os.getcwd())
import copy
import torch
from torch import autograd



def train_iters(networks,
                optimizers,
                latent_dim,
                loader,
                loader_gen,
                iters,
                critters,
                lamb,
                skip_last=False):
    """
    Will train WGAN until the generator has been updated the number of times specified by 'iters' (minus one if skip_last==True).
    Before each update of the generator, the critic is trained for a number of iterations specified by 'critters'. If skip_last
    is True, the generator is not updated after the last critic update (the next generator update should then be done using 
    'misalignment_iter').
    """
    # ------------------------------- Setting up -------------------------------
    generator = networks[0]
    critic = networks[1]
    optim_g = optimizers[0]
    optim_c = optimizers[1]
    
    use_cuda = next(generator.parameters()).is_cuda
    
     # ------------------------------- Training -------------------------------
    for iteration in range(iters):
        #####################
        # (1) Update critic #
        #####################
        # Reset all parameters of all critics to requires_grad=True. (They are set to false when unpdating the generators)
        for p in critic.parameters():  
            p.requires_grad = True 
        
        for i in range(critters):
            # Reset critic gradients
            for param in critic.parameters(): 
                param.grad = None   # more efficient than zero_grad()
            
            # Get minibatches of real and fake data. 
            try:
                real = loader_gen.next()[0]
            except StopIteration:
                loader_gen = iter(loader)
                real = loader_gen.next()[0]
            noise = torch.randn(loader.bs, latent_dim)
            if use_cuda:
                real = real.cuda()
                noise = noise.cuda()
            fake = generator(noise).detach() # We detach to avoid useless backpropagation through generator
            
            # Compute and backpropagate critic values on real and fake data
            critval_r = critic(real).mean()
            critval_r.backward()    
            critval_f = -critic(fake).mean()
            critval_f.backward()  
            
            # Compute and backpropagate gradient penalty
            grad_pen = grad_penalty(critic, real, fake, lamb)
            grad_pen.backward()
            
            # Optimizer step
            optim_c.step()
        
        ########################
        # (2) Update generator #
        ########################
        if iteration==iters-1 and skip_last:
            return loader_gen
    
        # Turn off grad computations for critic
        for p in critic.parameters():
            p.requires_grad = False  # to avoid computation
        
        # Reset generator gradient
        for param in generator.parameters():
            param.grad = None # more efficient than netG.zero_grad()
        
        # Get minibatch of noise inputs and generate fake data
        noise = torch.randn(loader.bs, latent_dim)
        if use_cuda:
            noise = noise.cuda()
        fake = generator(noise)
        
        # Compute and backpropagate critic value on fake data
        critval = critic(fake).mean()
        critval.backward()
        
        # Optimizer step
        optim_g.step()
    
    return loader_gen
        
            
            
def misalignment_iter(critic, generator, optimizer, noise1, noise2):
    """
    Computes and returns three types of update vectors for the generator at the generated samples corresponding to each
    noise input in noise1
    
    Inputs
        - critics: a critic network 
        - generators: a generator network
        - optimizer: the optimizer used to update the generator
        - noise1: A batch of noise inputs used to generate the fake data where the update vectors will be computed.
        - noise2: A batch of noise inputs used to update the generator.
    
    Outputs (each a torch tensor of size [noise1 x data_shape])
        - crit_gradients: The gradients of the critic at the generated samples. These are the directions in which
                          the generated samples would optimally move when updating the generator's parameters.
        - sgd_updates: Vectors in the directions along which the generated samples would move if the generator 
                       updates were done using basic SGD.
        - adam_updates: Actual updates of the generated samples obtained while training the generator with Adam.
    """
    # ------------------------------- Critic gradients ------------------------------------------------
    fake1 = generator(noise1).detach().clone().requires_grad_()
    crit_vals1 = critic(fake1)
    crit_grads = torch.autograd.grad(outputs=crit_vals1, 
                                     inputs=fake1,
                                     grad_outputs=torch.ones_like(crit_vals1),
                                     only_inputs=True)[0]
    
    # ------------------------------- Generator updates ------------------------------------------------
    # Create copy of generator which will have its weights changed to tensors
    gen_copy = copy.deepcopy(generator)
    params, names = extract_weights(gen_copy)
    
    # Define function for JVP:
    def gen_on_params_(*new_params):
        load_weights(gen_copy, names, new_params)
        return gen_copy(noise1)

    # compute direction vector for JVP
    crit_vals2 = torch.mean(critic(generator(noise2)))
    critic.zero_grad()
    generator.zero_grad()
    crit_vals2.backward()
    vector = []
    for param in generator.parameters():
        vector.append(param.grad.detach().clone())
    vector = tuple(vector)

    critic.zero_grad() # re-zero grads to avoid causing problems later
    
    # Compute JVP
    _, sgd_updates = torch.autograd.functional.jvp(gen_on_params_, params, v = vector)
    
    # ------------------------------- Optimizer updates ------------------------------------------------
    optimizer.step()
    fake1_new = generator(noise1)
    adam_updates = fake1 - fake1_new
    
    return crit_grads, sgd_updates, adam_updates



def grad_penalty(critic, fake, real, lamb):
    """
    Gradient penalty used by 'train_iters' to train the critic network.
    """
    bs = fake.shape[0]
    
    # Get points where gradient penalty will be computed
    t = torch.rand((bs,1,1,1))
    if fake.is_cuda:
        t = t.cuda()
    interpolates = (1-t)*fake + t*real
    
    # Compute gradient penalty at those points
    interpolates.requires_grad = True
    critvals_interpolates = critic(interpolates)
    gradients = autograd.grad(outputs=critvals_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(critvals_interpolates),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    grad_pen = (torch.clamp(gradients.norm(2, dim=1)-1,min =0)**2).mean() * lamb
    
    return grad_pen



# ------------------------------- Functions used to get SGD updates directions -------------------------------
def extract_weights(network):
    """
    Will return the current parameters of the network along with their names.
    Inputs:     - A neural network object (i.e. from torch.nn.Module)
    
    Outputs:    - A tuple containing the tensor in each parameter of the network at the moment the function was called.
                - A tuple containing the names of each parameter.
    """
    params = []
    names = []
    for name, param in network.named_parameters():
        params.append(param.detach().clone())
        names.append(name)
    params = tuple(params)
    names = tuple(names)
    return params, names
        
    
    
def load_weights(network, names, params):
    """
    Will change network parameters of the network to the specified values. 
    Inputs:     - A neural network object
                - A tuple containing the names of parameters that will be changed
                - A tuple containing tensors which will be the new values of the parameters to change.
    """
    # Delete all parameter attributes
    for name in names:
        del_param(network, name.split("."))
    
    # Recreate all parameter attributes as nn.tensor's 
    for i in range(len(names)):
        set_param(network, names[i].split("."), params[i])
    


def del_param(obj, attr_name):
    if len(attr_name)==1:
        delattr(obj, attr_name[0])
    else:
        del_param(getattr(obj, attr_name[0]), attr_name[1:])



def set_param(obj, attr_name, val):
    if len(attr_name)==1:
        setattr(obj, attr_name[0], val)
    else:
        set_param(getattr(obj, attr_name[0]), attr_name[1:], val)




