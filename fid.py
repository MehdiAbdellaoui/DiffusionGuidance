# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Script for calculating Frechet Inception Distance (FID)."""

import os
import click
import pickle
import numpy as np
import scipy.linalg
import torch
import dnnlib

#----------------------------------------------------------------------------


def calculate_inception_stats_npz(image_path, num_samples=50000, samples_per_batch=500, device=torch.device('cuda'),
):
    print('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=(0 == 0)) as f:
        detector_net = pickle.load(f).to(device)

    print(f'Loading images from "{image_path}"...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    # Load and shuffle images
    files = np.load(image_path)['images']
    count = 0
    num_batch_images = samples_per_batch
    num_batches = int(num_samples / num_batch_images)
    for i in range(num_batches):
        images = files[i * num_batch_images:(i + 1) * num_batch_images]
        images = torch.tensor(images).permute(0, 3, 1, 2).to(device)
        features = detector_net(images, **detector_kwargs).to(torch.float64)
        if count + images.shape[0] > num_samples:
            remaining_num_samples = num_samples - count
        else:
            remaining_num_samples = images.shape[0]
        mu += features[:remaining_num_samples].sum(0)
        sigma += features[:remaining_num_samples].T @ features[:remaining_num_samples]
        count = count + remaining_num_samples
        # Scuffed way of writing to the same line
        print('                               ', end='\r')
        print(f'Progress: {i / num_batches}%', end='\r')
        if count >= num_samples:
            break
    print('                               ')
    print(count)
    mu /= num_samples
    sigma -= mu.ger(mu) * num_samples
    sigma /= num_samples - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()


#----------------------------------------------------------------------------
def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))


#----------------------------------------------------------------------------
@click.command()
@click.option('--images', 'image_path', help='Path to the images', metavar='PATH|ZIP',              type=str, required=True)
@click.option('--ref', 'ref_path',      help='Dataset reference statistics ', metavar='NPZ|URL',    type=str, required=True)
@click.option('--num_samples', metavar='INT', default=50000)
@click.option('--samples_per_batch', metavar='INT', default=100)
@click.option('--device', metavar='STR', default='cuda')
def main(image_path, ref_path, num_samples, samples_per_batch, device):
    """Calculate FID for a given set of images."""
    print(f'Loading dataset reference statistics from "{ref_path}"...')
    with dnnlib.util.open_url(ref_path) as f:
        ref = dict(np.load(f))
    mu, sigma = calculate_inception_stats_npz(image_path=image_path, num_samples=num_samples,
                                              samples_per_batch=samples_per_batch, device=device)

    print('Calculating FID...')
    fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
    print(f'{image_path.split("/")[-1]}, {fid:g}')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------vd--------------------------------------------------------------------