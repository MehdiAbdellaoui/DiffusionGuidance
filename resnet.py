import torch
from torch import nn
from discriminator import get_discriminator_model, get_ADM_model
from guided_diffusion.script_util import create_classifier

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Embedding the class and timestep information
        self.embedding_layer = nn.Sequential(
            nn.Linear(
                emb_channels,
                self.out_channels,
            ),
            nn.SiLU(),
        )

        # Creating the two layered convolutional network
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.last_activation = nn.SiLU(out_channels)

        # In case downsampling is required
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.downsample = None

    def forward(self, x, emb):
        # Condition on timestep and class
        embedding = self.embedding_layer(emb)
        # Feed through convolutions
        output = self.conv1(x)
        output = self.conv2(output)
        # Down sample if needed
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = nn.Identity(x)
        # Add the skip layer and emebedding
        output = output + residual + embedding
        # Final activation function
        output = self.last_activation(output)
        return output


class DiscriminatorResNet(nn.Module):
    def __init__(self):
        super(DiscriminatorResNet, self).__init__()
        layer0 = 0


if __name__ == '__main__':
    # Testing for LoRA
    original_state_dict = torch.load('guided_diffusion/model/32x32_classifier.pt')
    adm_classifier = create_classifier(
        image_size=32,
        classifier_in_channels=3,
        classifier_out_channels=1000,
        classifier_width=128,
        classifier_depth=4,
        classifier_attention_resolutions="32,16,8",
        classifier_pool='attention',
        conditioned=False,
        lora_rank=4
    )
    # Create a new state dictionary with renamed keys
    new_state_dict = {}
    for key in original_state_dict.keys():
        new_key = key
        # Example of renaming logic (adjust according to your needs)
        if 'in_layers.2.weight' in key:
            new_key = key.replace('in_layers.2.weight', 'in_layers.2.conv.weight')
        elif 'in_layers.2.bias' in key:
            new_key = key.replace('in_layers.2.bias', 'in_layers.2.conv.bias')
        new_state_dict[new_key] = original_state_dict[key]

    missing_keys, unexpected_keys = adm_classifier.load_state_dict(new_state_dict, strict=True)


