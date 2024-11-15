import torch
from torch import nn

def check_grads(model: nn.Module):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if torch.isnan(parameter.grad).any():
                print(f"NaN gradient in {name}")
