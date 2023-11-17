from guided_diffusion.script_util import create_classifier
import torch
import torch.nn as nn
import numpy as np
import os


# Create the ADM model as explained in the paper and the given ADM checkpoint
def get_ADM_model():
    adm_classifier = create_classifier(
        image_size=32,
        classifier_in_channels=3,
        classifier_out_channels=1000,
        classifier_width=128,
        classifier_depth=4,
        classifier_attention_resolutions="32,16,8",
        classifier_pool='attention',
        conditioned=False
    )
    pretrained_adm = torch.load('guided_diffusion/model/32x32_classifier.pt')
    adm_classifier.load_state_dict(pretrained_adm)
    return adm_classifier


# Create the discriminator as given in the paper
def get_discriminator_model(conditioned):
    discriminator = create_classifier(
        image_size=8,
        classifier_in_channels=512,
        classifier_out_channels=1,
        classifier_width=128,
        classifier_depth=4,
        classifier_attention_resolutions="8",
        classifier_pool='attention',
        conditioned=conditioned
    )
    return discriminator


# Implemented based on page 24 and 15
class WVEtoLVP:
    """Class to convert between the training modes of EDM and ADM"""
    def __init__(self):
        self.beta_min = 0.1
        self.beta_max = 20.
        self.T = 1.     # For u sampled uniformly from [0, T]

    def transform_to_tau(self, var_wve_t):
        temp = self.beta_min ** 2 + 2. * (self.beta_max - self.beta_min) * torch.log(1. + var_wve_t)
        tau = (-self.beta_min + torch.sqrt(temp)) / (self.beta_max - self.beta_min)
        return tau

    def marginal_prob(self, t):
        mean_coef = -t ** 2 * (self.beta_max - self.beta_min) / 4 - t * self.beta_min / 2
        mean = torch.exp(mean_coef)
        std = torch.sqrt(1. - torch.exp(2. * mean_coef))
        return mean, std

    def transform_WVE_to_LVP(self, std_wve_t):
        tau = self.transform_to_tau(std_wve_t ** 2)
        mean, _ = self.marginal_prob(tau)
        return mean, tau

    def antiderivative(self, t):
        if isinstance(t, float) or isinstance(t, int):
            t = torch.tensor(t).float()
        integral_beta = 0.5 * t ** 2 * (self.beta_max - self.beta_min) + t * self.beta_min
        return torch.log(1. - torch.exp(- integral_beta)) + integral_beta

    def generate_diffusion_times(self, n_samples, device, t_min=1e-5):
        # See page 15 for derivations of importance sampler!
        # Calculate F(t) = u * Z - F(t_min), where F(t) is the antiderivative of the importance weight
        anti_t_min = self.antiderivative(t_min)
        Z = self.antiderivative(self.T) - anti_t_min
        u = torch.rand(n_samples, device=device) * Z + anti_t_min
        return self.transform_to_tau(torch.exp(u))


if __name__ == '__main__':
    get_ADM_model()
    get_discriminator_model(True)
