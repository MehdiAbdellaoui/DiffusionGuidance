import matplotlib.pyplot as plt
import numpy as np


# Quick helper function to combine npz files
def combine_npz(file_names, num_batches, save_name, conditional):
    images = None
    labels = None
    batch = 0
    for i, files in enumerate(file_names):
        file = np.load(files)
        if images is None or labels is None:
            images = file['images']
            if conditional:
                labels = file['labels']
        else:
            images = np.concatenate((images, file['images']))
            if conditional:
                labels = np.concatenate((labels, file['labels']))

        if i % (len(file_names) / num_batches) == (len(file_names) / num_batches) - 1:
            if conditional:
                np.savez_compressed(save_name + str(batch), images=images, labels=labels)
            else:
                np.savez_compressed(save_name + str(batch), images=images)
            images = None
            labels = None
            batch += 1


if __name__ == '__main__':
    combine = False
    if combine is True:
        # Combine into 5 batches with 10000 images and labels in each batch
        num_files = 200
        file_names = ['training_data/cond_samples_batch' + str(i) + '.npz' for i in range(num_files)]
        save_name = 'training_data/cond_samples'
        combine_npz(file_names, 5, save_name, True)
    else:
        # Check such that the combined batches are not duplicates
        file1 = np.load('training_data/cond_samples1.npz')
        file2 = np.load('training_data/cond_samples2.npz')
        plt.imshow(file1['images'][10])
        plt.show()
        plt.imshow(file2['images'][10])
        plt.show()
