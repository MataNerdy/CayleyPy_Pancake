from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Dict, Iterable, List, Any

BASE_CFG: Dict = dict(
    rw_width=3000,
    rw_mode="mix",
    mix_bfs_frac=0.3,
    nbt_history_depth=1,
    k=4,
    lr=1e-4,
    weight_decay=1e-2,
    batch_size=1024,
    val_ratio=0.15,
    num_epochs=30,
    y_transform="log1p",
    grad_clip=1.0,
    loss_beta=1.0,
    stratify_clip=60,
    stratify_bin_size=1.0,
    sanity_width=2000,
    h_batch_size=8192,
    eval_log_every=10,
    seed=123,
    rw_length_add=30,
    early_stop_patience=6,
    early_stop_min_delta=1e-4,
    early_stop_warmup=3,
    early_stop_restore_best=True,
)

def make_cfg(**overrides) -> Dict:
    cfg = deepcopy(BASE_CFG)
    cfg.update(overrides)
    return cfg

def build_model_grid(
    *,
    emb_dims: Iterable[int] = (32, 64),
    pos_opts: Iterable[bool] = (True, False),
    nrds: Iterable[int] = (0, 2, 6),
    hd1: int = 512,
    hd2: int = 256,
    dropout_rate: float = 0.1,
    include_mlpres1: bool = True,
) -> List[Dict]:
    grid: List[Dict] = []

    for ed, pos, nrd in product(emb_dims, pos_opts, nrds):
        grid.append(make_cfg(
            model_type="EmbMLP",
            emb_dim=int(ed),
            use_pos_emb=bool(pos),
            hd1=hd1, hd2=hd2, nrd=int(nrd),
            dropout_rate=float(dropout_rate),
        ))

    if include_mlpres1:
        for nrd in nrds:
            grid.append(make_cfg(
                model_type="MLPRes1",
                hd1=hd1, hd2=hd2, nrd=int(nrd),
                dropout_rate=float(dropout_rate),
            ))
    return grid



def build_leader_train_grid(
    *,
    leader_arch: Dict[str, Any],
    train_base: Dict[str, Any],
    rw_width_grid: Iterable[int] = (2000, 3000, 5000),
    mix_frac_grid: Iterable[float] = (0.15, 0.30, 0.50),
    lr_grid: Iterable[float] = (1e-4, 2e-4),
    weight_decay_grid: Iterable[float] = (5e-3, 1e-2),
    epochs_grid: Iterable[int] = (40,),
) -> List[Dict[str, Any]]:
    model_grid: List[Dict[str, Any]] = []
    for rw, mf, lr, wd, ep in product(rw_width_grid, mix_frac_grid, lr_grid, weight_decay_grid, epochs_grid):
        cfg = {}
        cfg.update(leader_arch)
        cfg.update(train_base)
        cfg.update(dict(
            rw_width=int(rw),
            mix_bfs_frac=float(mf),
            lr=float(lr),
            weight_decay=float(wd),
            num_epochs=int(ep),
        ))
        model_grid.append(cfg)
    return model_grid
