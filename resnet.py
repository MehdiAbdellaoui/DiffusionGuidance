import torch
from torch import nn
from discriminator import get_discriminator_model, get_ADM_model
from guided_diffusion.script_util import create_classifier
from guided_diffusion.unet import AttentionPool2d


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
            nn.SiLU()
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
    def __init__(self, input_channels):
        super(DiscriminatorResNet, self).__init__()
        upscale = input_channels * 2
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.num_classes = num_classes
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        self.resnet = nn.Sequential(
            ResidualBlock(input_channels, input_channels, 1000),
            ResidualBlock(input_channels, input_channels, 1000),
            ResidualBlock(input_channels, input_channels, 1000),
            ResidualBlock(input_channels, upscale, 1000),
            ResidualBlock(upscale, upscale, 1000),
            ResidualBlock(upscale, upscale, 1000),
        )
        self.pooling = nn.Sequential(
                nn.BatchNorm2d(upscale),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(upscale, 1),
                nn.Sigmoid()
                )

    def _forward(self, x, timesteps, labels):
        timesteps = timesteps * 999.
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            emb = emb + self.label_emb(torch.argmax(labels, 1))
        for block in self.resnet:
            x = block(x, emb)
        x = self.pooling(x)
        return x

