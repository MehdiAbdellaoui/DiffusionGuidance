import matplotlib.pyplot as plt
import numpy as np
import pickle


# Quick helper function to combine npz files
def combine_npz(file_names, num_batches, save_name, conditional):
    images = None
    labels = None
    batch = 0
    for i, files in enumerate(file_names):
        file = np.load(files)
        if images is None or labels is None:
            images = file['data']
            labels = 0
            if conditional:
                labels = file['labels']
        else:
            images = np.concatenate((images, file['data']))
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
    combine = True
    if combine is True:
        # Combine into 5 batches with 10000 images and labels in each batch
        num_files = 5
        file_names = ['training_data/data_batch_' + str(i) for i in range(num_files)]
        save_name = 'training_data/edm_uncond_samples'
        combine_npz(file_names, 1, save_name, True)
    else:
        # Check such that the combined batches are not duplicates
        file = np.load('training_data/edm_uncond_samples0.npz')
        print(file['images'].shape)
        plt.imshow(file['images'][0])
        plt.show()
        plt.imshow(file['images'][250])
        plt.show()
