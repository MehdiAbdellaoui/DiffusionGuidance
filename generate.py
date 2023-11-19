# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist

import random
import math
import discriminator as discriminator_lib

#----------------------------------------------------------------------------
# Proposed EDM-G++ sampler 

def edm_sampler(
    discriminator, w_DG_1, w_DG_2, time_mid, time_max, adaptive_weight, vpsde,
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    S_churn_manual = 4
    period_weight = 2 # for every odd denoising steps (page 22 of paper)
    period_churn = 2 # 5 in their implementation? for every odd denoising steps (page 22 of paper)

    # define vector of length equal to number of samples to be able to modify parameter values
    # depending on density-ratio of specific sample (see page 22 of paper)
    S_churn_vector_default = torch.tensor([S_churn] * latents.shape[0], device=latents.device)
    S_churn_vector_max = torch.tensor([np.sqrt(2) - 1] * latents.shape[0], device=latents.device)
    S_noise_vector = torch.tensor([S_noise] * latents.shape[0], device=latents.device)
    
    # set to large positive value by default
    log_ratio = torch.tensor([np.finfo(np.float64).max] * latents.shape[0], device=latents.device)

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        
        S_churn_vector = S_churn_vector_default.clone()
         
        if adaptive_weight:
            # for every odd denoising steps (page 22 of paper)
            if i % period_churn == 0: # consider start at step 1
                # for such samples with density-ratio less than 0 (page 22 of paper)
                S_churn_vector[log_ratio < 0.] = S_churn_manual

        # Increase noise temporarily.
        # adapted due to vectorized version of S_churn
        gamma = torch.minimum(S_churn_vector / num_steps, S_churn_vector_max) if S_min <= t_cur <= S_max else torch.zeros_like(S_churn_vector)
        
        # sigma(t) = t from EDM paper
        t_hat = net.round_sigma(t_cur + gamma * t_cur) 
        
        # adapted due to vectorized version of t_hat and S_noise 
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt().reshape(x_cur.shape) * S_noise_vector.reshape(x_cur.shape) * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        
        # adapted due to vectorized version of t_hat
        d_cur = (x_hat - denoised) / t_hat.reshape(x_hat.shape)
        
        # first order correction according to Heun solve (see page 21 of paper)
        if w_DG_1 != 0.:

            discriminator_output, log_ratio = discriminator_lib.get_gradient_density_ratio(discriminator, vpsde, x_hat, t_hat, time_mid, time_max, img_size, class_labels)
            
            if adaptive_weight:
                # for every odd denoising steps (page 22 of paper)
                if i % period_weight == 0:
                    # for such samples with density-ratio less than 0 (page 22 of paper)
                    # instead of setting w_DG_1 = 2 instead of w_DG_1 = 1 in this case
                    # we equivalently operate on the output of the discriminator
                    discriminator_output[log_ratio < 0.] *= 2 
        
            # adjust d_cur with discriminator output and divide to make it compatible with previous d_cur
            d_cur += w_DG_1 * (discriminator_output / t_hat.reshape(x_hat.shape))

        # adapted due to vectorized version of t_hat
        x_next = x_hat + (t_next - t_hat).reshape(x_hat.shape) * d_cur
        
        # Apply 2nd order correction.
        if i < num_steps - 1: # otherwise t_next is not available
            denoised = net(x_next, t_next, class_labels).to(torch.float64)

            # no need to adapt because t_next is not vectorized
            d_prime = (x_next - denoised) / t_next
            
            # second order correction according to Heun solve (see page 21 of paper)
            if w_DG_2 != 0.:
                
                discriminator_output, _ = discriminator_lib.get_gradient_density_ratio(discriminator, vpsde, x_hat, t_hat, time_mid, time_max, img_size, class_labels)
            
                # adjust d_prime with discriminator output and divide to make it compatible with previous d_cur
                d_prime += w_DG_2 * (discriminator_output / t_next)

            # adapted due to vectorized version of t_hat
            x_next = x_hat + (t_next - t_hat).reshape(x_hat.shape) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
#@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
#@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

#----------------------------------------------------------------------------
# Discriminator Guidance
#----------------------------------------------------------------------------

# Configuration
# Default values come from Figure 21 of page 22
@click.option('--w_DG_1',                     help='Weight for 1st order DG', metavar='FLOAT',                              type=click.FloatRange(min=0), default=2, show_default=True)
@click.option('--w_DG_2',                     help='Weight for 2nd order DG', metavar='FLOAT',                              type=click.FloatRange(min=0), default=0, show_default=True)

# Discriminator checkpoint and architecture
@click.option('--discriminator_checkpoint',   help='Path to discriminator checkpoint', metavar='STR',                       type=str, default='', show_default=True)
@click.option('--conditional',   help='Conditional discriminator?', is_flag=True)

# Sampling configuration
@click.option('--seed', help='Seed value', metavar='INT',                                type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--num_samples', help='Number of samples to generates', metavar='INT',                                type=click.IntRange(min=1), default=50000, show_default=True)
@click.option('--save_format',   help='Format for storing the generated samples', metavar='png|npz',                       type=click.Choice(['png', 'npz']), default='npz', show_default=True)
@click.option('--device',   help='Device', metavar='STR',                       type=str, default='cuda:0', show_default=True)

# denoise with the reverse-time generative process that includes the discriminator in the range [time_mid, time_max] (pages 26 and 27)
# default values come from Table 8 page 21
@click.option('--time_mid', help='Start time for applying the discriminator', metavar='FLOAT', type=click.FloatRange(0, 1), default=0.01, show_default=True)
@click.option('--time_max', help='End time for applying the discriminator', metavar='FLOAT', type=click.FloatRange(0, 1), default=1, show_default=True)

# Adaptive strategy is only applied to conditional generation as indicated in Table 8 page 21
@click.option('--adaptive_weight',   help='Enable adaptive strategy to boost the DG weights?', is_flag=True)

def main(w_DG_1, w_DG_2, discriminator_checkpoint, conditional, seed, num_samples, save_format, time_mid, time_max, adaptive_weight,
    network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, device, **sampler_kwargs):
    
    # Set manual seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set device
    device = torch.device(device)

    # Load pretained score network.
    print(f'Loading network from "{network_pkl}"...')
    with open(network_pkl, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)
    
    # Load pretained discriminator network.
    if w_DG_1 != 0 or w_DG_2 != 0:
        print(f'Loading discriminator from "{discriminator_checkpoint}"...')
        discriminator = discriminator_lib.get_discriminator(discriminator_checkpoint, device, conditioned=(net.label_dim and conditional)) 
    else:
        discriminator = None
    
    # Variance Preserving SDE (VPSDE) introduced in "Score-based generative modeling through stochastic differential equations"
    # Page 25: while score training is beneficial with WVE-SDE, discriminator training best fits with LVP-SDE. We, therefore, train the discriminator with LVP-SDE as default    
    vpsde = discriminator_lib.WVEtoLVP() 

    # Loop over batches.
    num_batches = math.ceil(num_samples / max_batch_size)
    
    print(f'Generating {num_samples} images to "{outdir}"...')
    
    for i in tqdm.tqdm(range(num_batches)):

        # Pick latents and labels.
        latents = torch.randn([max_batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        
        class_labels = None
        
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[max_batch_size], device=device)]
        
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        images = edm_sampler(discriminator, w_DG_1, w_DG_2, time_mid, time_max, adaptive_weight, vpsde, net, latents, class_labels, randn_like=torch.randn_like, **sampler_kwargs)

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        if save_format == 'png':
            for idx, image_np in enumerate(images_np):
                index = i * max_batch_size + idx
                image_path = os.path.join(outdir, f'{index:06d}.png')
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)
        elif save_format == 'npz':
            if class_labels is None:
                image_path = os.path.join(outdir, f'unconditional_{i:06d}.npz')
                #np.savez_compressed('training_data/uncond_samples_batch' + str(cur_batch) + '.npz', images=images_np)
            else:
                image_path = os.path.join(outdir, f'conditional_{i:06d}.npz')
                #np.savez_compressed('training_data/cond_samples_batch' + str(cur_batch) + '.npz', images=images_np,
                #                    labels=class_labels.cpu().numpy())
            
            np.savez_compressed(image_path, images=images_np)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
