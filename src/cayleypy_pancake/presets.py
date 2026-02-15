from __future__ import annotations

from typing import Dict, Any


DEFAULT_BEAM = dict(
    beam_width=256,
    depth=192,
    w=0.5,
    w_gap=0.15,
    gap_mode="log1p",
    h_batch_size=8192,
)

LEADER_CFG: Dict[str, Any] = dict(
    model_type="EmbMLP",
    emb_dim=32,
    use_pos_emb=False,
    hd1=512, hd2=256, nrd=6, dropout_rate=0.1,

    rw_width=5000,
    rw_mode="mix",
    mix_bfs_frac=0.30,
    nbt_history_depth=1,

    lr=2e-4,
    weight_decay=1e-2,
    batch_size=1024,
    val_ratio=0.15,
    num_epochs=30,

    y_transform="log1p",
    grad_clip=1.0,
    loss_beta=1.0,

    sanity_width=2000,
    h_batch_size=8192,
    eval_log_every=200,
    seed=123,

    stratify_clip=60,
    stratify_bin_size=1.0,
    k=4,
)
