
import os
import pickle
import numpy as np
import torch
import PIL.Image
import tensorflow as tf
import io
from torchvision.utils import make_grid, save_image

images_np = np.load('uncond_lora128.npz')['images']
#images_np = np.load('./training_data/unconditional_edm_samples/edm_uncond_samples.npz')['images']
images_np = images_np[:100]

nrow = int(np.sqrt(images_np.shape[0]))
image_grid = make_grid(torch.tensor(images_np).permute(0, 3, 1, 2) / 255., nrow, padding=2)

with tf.io.gfile.GFile(os.path.join('output_images', f"sample_grid.png"), "wb") as fout:
	save_image(image_grid, fout)
