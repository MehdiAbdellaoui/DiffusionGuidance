# Reproducing the results from Discriminator Guidance in Score-based Diffusion Models

## Authors

- Mehdi Abdellaoui
- Carl Persson 
- Eduardo Santos Escriche

## Overview

In this repository we provide the code for our attempt at reproducing the results from [Discriminator Guidance in Score-based Diffusion Models](https://arxiv.org/abs/2211.17091), where the authors introduce a new method aiming to improve the sample generation process for pre-trained diffusion models, called Discriminator Guidance

In our [project report](https://github.com/MehdiAbdellaoui/DiffusionGuidance/blob/main/report/DD2412_Project_Report.pdf), we are able to produce very similar FID scores to those presented in the paper for the CIFAR-10 dataset and the EDM-G++ sampling method. In addition, we explore the statistical significance of those results, as well as the sensitivity of the $S_{churn}$ hyperparameter. Lastly, we go beyond the scope of the paper by implementing and analyzing the performance of including the Low-Rank adaptation (LoRA) method for finetuning the discriminator, which we show can improve the FID score for unconditionally generated samples from $1.83$ to $1.79$.

## Code execution

### 1) Create conda environment

```
conda create --name <env> --file requirements.txt
```

### 2) Prepare a pre-trained score network

### 3) Generate fake samples

### 4) Prepare real data

### 5) Prepare a pre-trained classifier

- Download [DG/checkpoints/ADM_classifier/32x32_classifier.pt](https://drive.google.com/drive/folders/1gb68C13-QOt8yA6ZnnS6G5pVIlPO7j_y)
- Place **32x32_classifier.pt** at the directory specified below.

### 6) Train a discriminator

### 7) Generate discriminator-guided samples

### 8) Evaluate FID

## Experimental Results

### EDM-G++

### Samples from unconditional Cifar-10 EDM 

![sample_grid_lora](./plots/sample_grid_edm.png)

### Samples from unconditional Cifar-10 EDM-G++ with LoRA

![sample_grid_lora](./plots/sample_grid.png)

## Additional references

Similarly to the original repository, we also take inspiration from the methods proposed in the following papers:

- *Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. arXiv preprint arXiv:2206.00364.*
- *Dhariwal, P., & Nichol, A. (2021). Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34, 8780-8794.*
- *Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020). Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456.*



