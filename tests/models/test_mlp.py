#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pytest test file for the MLP model."""

import pytest
import torch

from models.mlp import MLP

# ---- Test Constants ----
INPUT_DIM = 784  # Example: 28x28 images flattened
HIDDEN_DIM = 128
OUTPUT_DIM = 10  # Example: 10 classes for digits 0-9
BATCH_SIZE = 32


# ---- Pytest Fixture ----
@pytest.fixture
def mlp_model() -> MLP:
    """Provides a reusable instance of the MLP model for tests."""
    # Set a fixed seed for reproducible test results.
    torch.manual_seed(42)
    return MLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)


# ---- Test Cases ----

def test_mlp_initialization(mlp_model: MLP):
    """Tests if the MLP model layers are initialized with correct dimensions."""
    # Check the first linear layer's input and output dimensions.
    first_linear_layer = mlp_model.layers[0]
    assert isinstance(first_linear_layer, torch.nn.Linear)
    assert first_linear_layer.in_features == INPUT_DIM
    assert first_linear_layer.out_features == HIDDEN_DIM

    # Check the second linear layer (located after ReLU and Dropout).
    second_linear_layer = mlp_model.layers[3]
    assert isinstance(second_linear_layer, torch.nn.Linear)
    assert second_linear_layer.in_features == HIDDEN_DIM
    assert second_linear_layer.out_features == OUTPUT_DIM


def test_forward_pass_output_shape(mlp_model: MLP):
    """Tests if the forward pass produces an output tensor of the correct shape."""
    # Create a standard 2D input tensor (batch_size, num_features).
    input_tensor = torch.randn(BATCH_SIZE, INPUT_DIM)

    # Pass the input through the model.
    output = mlp_model(input_tensor)

    # Verify the output tensor's shape and type.
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
    assert isinstance(output, torch.Tensor)


def test_forward_pass_with_flattening(mlp_model: MLP):
    """Tests if the model correctly flattens multi-dimensional input tensors."""
    # Create a 3D input tensor simulating a batch of 28x28 images.
    # Note: 28 * 28 must equal INPUT_DIM for this test.
    assert 28 * 28 == INPUT_DIM
    input_tensor_3d = torch.randn(BATCH_SIZE, 28, 28)

    # Pass the 3D tensor through the model.
    output = mlp_model(input_tensor_3d)

    # The output shape should still be correct, as the model should flatten the input.
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM)


def test_dropout_behavior_in_train_and_eval_modes(mlp_model: MLP):
    """Tests the Dropout layer's behavior in training vs. evaluation modes."""
    input_tensor = torch.randn(BATCH_SIZE, INPUT_DIM)

    # --- Test training mode ---
    # In train mode, the dropout layer is active.
    # Two forward passes with the same input should yield different results.
    mlp_model.train()
    output1_train = mlp_model(input_tensor)
    output2_train = mlp_model(input_tensor)
    assert not torch.equal(output1_train, output2_train)

    # --- Test evaluation mode ---
    # In eval mode, the dropout layer is disabled.
    # Two forward passes with the same input should yield identical results.
    mlp_model.eval()
    output1_eval = mlp_model(input_tensor)
    output2_eval = mlp_model(input_tensor)
    assert torch.equal(output1_eval, output2_eval)