import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden=(64, 32)):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def get_flat(model) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()]).clone()


def set_flat(model, vec: torch.Tensor):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[i:i + n].view_as(p))
        i += n
