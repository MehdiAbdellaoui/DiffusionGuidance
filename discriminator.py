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


if __name__ == '__main__':
    get_ADM_model()
    get_discriminator_model(True)
