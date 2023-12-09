import torch
from torch import nn


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


