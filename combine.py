import numpy as np
import matplotlib.pyplot as plt
from fid import *

# Quick helper function to combine npz files
def combine_npz(file_names, save_name, conditional):
    images = None
    labels = None
    for i, files in enumerate(file_names):
        file = np.load(files + '.npz')
        if images is None or labels is None:
            images = file['images']
            labels = 0
            if conditional:
                labels = file['labels']
        else:
            images = np.concatenate((images, file['images']))
            if conditional:
                labels = np.concatenate((labels, file['labels']))

    if conditional:
        np.savez_compressed(save_name, images=images, labels=labels)
    else:
        np.savez_compressed(save_name, images=images)


def plot_schurn():
    with dnnlib.util.open_url('training_data/CIFAR_ref/cifar10-FID-stats.npz') as f:
        ref = dict(np.load(f))

    schurn_numbers = ['1', '1,25', '1,5', '1,75', '2', '2,25', '2,5', '2,75', '3']
    files = ['training_data/schurn/schurn' + i + '.npz' for i in schurn_numbers]
    schurn = [1, 1.25, 1.50, 1.75, 2, 2.25, 2.50, 2.75, 3]
    fid_list = []
    for path in files:
        mu, sigma = calculate_inception_stats_npz(image_path=path, num_samples=10000,
                                                  samples_per_batch=100, device='cuda')
        fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
        fid_list.append(fid)

    plt.plot(schurn, fid_list, linestyle='--', marker='o')
    plt.xlabel(r'$S_{churn}$')
    plt.ylabel('FID-10k')
    plt.title(r'FID-10k score as a function of $S_{churn}$')
    plt.show()


if __name__ == '__main__':
    plot_schurn()

