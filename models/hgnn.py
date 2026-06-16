from __future__ import annotations

import torch
from torch import nn


class HGNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        x_proj = self.linear(x)
        dv = torch.sum(h, dim=1)
        de = torch.sum(h, dim=0)
        dv_inv_sqrt = torch.pow(dv + eps, -0.5)
        de_inv = torch.pow(de + eps, -1.0)
        dv_mat = torch.diag(dv_inv_sqrt)
        de_mat = torch.diag(de_inv)
        aggregate = dv_mat @ h @ de_mat @ h.t() @ dv_mat
        out = aggregate @ x_proj
        out = self.activation(out)
        return self.dropout(out)


class HGNNStack(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([HGNNLayer(dims[i], dims[i + 1], dropout=dropout) for i in range(num_layers)])

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out, h)
        return out
