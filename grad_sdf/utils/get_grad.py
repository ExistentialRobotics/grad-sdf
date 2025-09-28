import torch
from torch import autograd


def get_gradients(inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """
    Compute gradients of the outputs with respect to the inputs.
    Args:
        inputs: (n_points, 3) input points
        outputs: (n_points, ...) output values
    Returns:
        gradients: (n_points, 3) gradients of the outputs with respect to the inputs
    """
    gradients = autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
    return gradients
