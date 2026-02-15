from __future__ import annotations

from typing import Any, Dict, List
import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = x + r
        x = F.relu(x)
        return x


class Pilgrim(nn.Module):
    """
    One-hot MLP + optional 2nd layer + residual stack.
    Input: z (B,n) int permutation
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.dtype = torch.float32
        self.state_size = int(cfg["state_size"])
        self.num_classes = int(cfg["num_classes"])
        self.hd1 = int(cfg.get("hd1", 512))
        self.hd2 = int(cfg.get("hd2", 256))
        self.nrd = int(cfg.get("nrd", 0))
        self.dropout_rate = float(cfg.get("dropout_rate", 0.1))
        self.z_add = 0

        in_dim = self.state_size * self.num_classes
        self.input_layer = nn.Linear(in_dim, self.hd1)
        self.bn1 = nn.BatchNorm1d(self.hd1)
        self.drop1 = nn.Dropout(self.dropout_rate)

        if self.hd2 > 0:
            self.hidden_layer = nn.Linear(self.hd1, self.hd2)
            self.bn2 = nn.BatchNorm1d(self.hd2)
            self.drop2 = nn.Dropout(self.dropout_rate)
            hid = self.hd2
        else:
            self.hidden_layer = None
            self.bn2 = None
            self.drop2 = None
            hid = self.hd1

        self.residual_blocks = None
        if self.nrd > 0:
            self.residual_blocks = nn.ModuleList([ResidualBlock(hid, self.dropout_rate) for _ in range(self.nrd)])

        self.out = nn.Linear(hid, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.one_hot((z.long() + self.z_add), num_classes=self.num_classes).view(z.size(0), -1).to(self.dtype)
        x = self.input_layer(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)

        if self.hidden_layer is not None:
            x = self.hidden_layer(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.drop2(x)

        if self.residual_blocks is not None:
            for blk in self.residual_blocks:
                x = blk(x)

        return self.out(x).flatten()


class SimpleMLP(nn.Module):
    """
    Configurable one-hot MLP.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.dtype = torch.float32
        self.state_size = int(cfg["state_size"])
        self.num_classes = int(cfg["num_classes"])
        self.z_add = 0

        layers = list(cfg["layers"])
        batch_norms = list(cfg.get("batch_norms", [True] * len(layers)))
        dropouts = cfg.get("dropout_rates", 0.1)
        activations = cfg.get("activations", nn.ReLU())

        if not isinstance(dropouts, list):
            dropouts = [dropouts] * len(layers)
        if not isinstance(activations, list):
            activations = [activations] * len(layers)

        in_dim = self.state_size * self.num_classes
        seq: List[nn.Module] = []
        for h, bn, act, dr in zip(layers, batch_norms, activations, dropouts):
            seq.append(nn.Linear(in_dim, int(h)))
            if bn:
                seq.append(nn.BatchNorm1d(int(h)))
            seq.append(act)
            seq.append(nn.Dropout(float(dr)))
            in_dim = int(h)
        seq.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*seq)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.one_hot((z.long() + self.z_add), num_classes=self.num_classes).view(z.size(0), -1).to(self.dtype)
        return self.net(x).flatten()


class EmbMLP(nn.Module):
    """
    Embedding-based regressor (recommended).
    Input: z (B,n) ints in [0..n-1]
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.dtype = torch.float32
        self.state_size = int(cfg["state_size"])
        self.num_classes = int(cfg["num_classes"])
        assert self.state_size == self.num_classes, "For pancakes, state_size==num_classes==n"
        self.z_add = 0

        d = int(cfg.get("emb_dim", 32))
        self.use_pos_emb = bool(cfg.get("use_pos_emb", True))
        dropout = float(cfg.get("dropout_rate", 0.1))

        self.token_emb = nn.Embedding(self.num_classes, d)
        if self.use_pos_emb:
            self.pos_emb = nn.Embedding(self.state_size, d)
            self.register_buffer("_pos_idx", torch.arange(self.state_size, dtype=torch.long), persistent=False)

        hd1 = int(cfg.get("hd1", 512))
        hd2 = int(cfg.get("hd2", 256))
        nrd = int(cfg.get("nrd", 0))

        in_dim = self.state_size * d
        self.fc1 = nn.Linear(in_dim, hd1)
        self.bn1 = nn.BatchNorm1d(hd1)
        self.drop1 = nn.Dropout(dropout)

        if hd2 > 0:
            self.fc2 = nn.Linear(hd1, hd2)
            self.bn2 = nn.BatchNorm1d(hd2)
            self.drop2 = nn.Dropout(dropout)
            hid = hd2
        else:
            self.fc2 = None
            self.bn2 = None
            self.drop2 = None
            hid = hd1

        self.residual_blocks = None
        if nrd > 0:
            self.residual_blocks = nn.ModuleList([ResidualBlock(hid, dropout) for _ in range(nrd)])

        self.out = nn.Linear(hid, 1)

        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        if self.use_pos_emb:
            nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = (z.long() + self.z_add).clamp(min=0, max=self.num_classes - 1)
        x = self.token_emb(z)  # (B,n,d)
        if self.use_pos_emb:
            pos = self._pos_idx.to(z.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = x.reshape(z.size(0), -1).to(self.dtype)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)

        if self.fc2 is not None:
            x = self.fc2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.drop2(x)
        if self.residual_blocks is not None:
            for blk in self.residual_blocks:
                x = blk(x)
        return self.out(x).flatten()


def get_model(cfg: Dict[str, Any]) -> nn.Module:
    mt = cfg.get("model_type", "EmbMLP")
    if mt == "EmbMLP":
        return EmbMLP(cfg)
    if mt == "MLPRes1":
        return Pilgrim(cfg)
    if mt == "MLP":
        return SimpleMLP(cfg)
    raise ValueError(f"Unknown model_type={mt}")
