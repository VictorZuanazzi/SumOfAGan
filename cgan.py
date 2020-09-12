"""Implementation of conditional GAN by Victor Zuanazzi.
Code inspered by https://github.com/eriklindernoren/PyTorch-GAN"""

import numpy as np

import torch
import torch.nn as nn
from torch.tensor import Tensor

from typing import NoReturn, Tuple


class GeneratorMLP(nn.Module):
    """Conditional Generator implemented with an MLP."""
    def __init__(self, n_classes: int,
                 latent_dim: int,
                 img_shape: Tuple[int, int]) -> NoReturn:
        """Constructor of the Generator.

        Args:
            n_classes: number of classes.
            latent_dim: size of the input latent dim (the noise input)
            img_shape: size of the output image.
        """
        super().__init__()
        self.img_shape = img_shape

        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = nn.ModuleList([nn.Linear(in_feat, out_feat)])
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise: Tensor, labels: int) -> \
            Tensor:
        """Forward pass of the generator.

        Args:
            noise: noise vector.
            labels: the class of the image that will be generated.

        Returns:
            An image generated from the label and noise.
        """
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(-1, *self.img_shape)
        return img


class DiscriminatorMLP(nn.Module):
    """Conditional Discriminator implemented with an MPL."""
    def __init__(self, n_classes: int, img_shape: Tuple[int, int]) -> NoReturn:
        """Constructor of the Discriminator.

        Args:
            n_classes: number of classes.
            img_shape: the size of the input images.
        """
        super().__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img: Tensor, labels: int):
        """Forward pass of the Discriminator.

        Args:
            img: the image that should be classified in fake or real.
            labels: the label of the image.

        Returns:

        """
        d_in = torch.cat((img.view(img.size(0), -1),
                          self.label_embedding(labels)), -1)
        score = self.model(d_in)
        return score
