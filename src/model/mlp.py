#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Defines a simple Multi-Layer Perceptron (MLP) model using PyTorch."""

import torch
from torch import nn


class MLP(nn.Module):
    """A simple Multi-Layer Perceptron with one hidden layer.

    This model consists of an input layer, a ReLU activation with dropout,
    and an output layer. It is designed to output raw logits.

    Attributes:
        layers: A sequential container of the network layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initializes the MLP model.

        Args:
            input_dim: The dimension of the input features.
            hidden_dim: The number of neurons in the hidden layer.
            output_dim: The dimension of the output (number of classes).
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the model.

        Args:
            x: The input tensor. It will be flattened before being passed
               through the network.

        Returns:
            The output tensor containing the raw logits for each class.
        """
        # Flatten the input tensor to a 2D shape [batch_size, num_features]
        x = x.view(x.size(0), -1)
        return self.layers(x)
