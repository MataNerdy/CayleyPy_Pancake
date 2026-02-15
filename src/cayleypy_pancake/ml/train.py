from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from cayleypy_pancake.ml.data import gen_walks, y_transform_torch, make_bins_np


def train_model_gpu(cfg: dict, model: torch.nn.Module, graph) -> None:
    device = graph.device
    model.to(device)

    bs = int(cfg["batch_size"])
    val_ratio = float(cfg["val_ratio"])
    epochs = int(cfg["num_epochs"])

    y_mode = cfg.get("y_transform", "log1p")
    grad_clip = float(cfg.get("grad_clip", 1.0))
    weight_decay = float(cfg.get("weight_decay", 1e-2))
    loss_beta = float(cfg.get("loss_beta", 1.0))

    strat_clip = int(cfg.get("stratify_clip", 60))
    strat_bin_size = float(cfg.get("stratify_bin_size", 1.0))

    rw_mode = cfg.get("rw_mode", "nbt")
    mix_bfs_frac = float(cfg.get("mix_bfs_frac", 0.3))
    es_patience = int(cfg.get("early_stop_patience", 0))
    es_min_delta = float(cfg.get("early_stop_min_delta", 0.0))
    es_warmup = int(cfg.get("early_stop_warmup", 0))
    es_restore = bool(cfg.get("early_stop_restore_best", True))

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=weight_decay)
    loss_fn = torch.nn.SmoothL1Loss(beta=loss_beta)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()

        if rw_mode == "mix":
            w_total = int(cfg["rw_width"])
            w_bfs = max(1, int(round(w_total * mix_bfs_frac)))
            w_nbt = max(1, w_total - w_bfs)

            cfg_bfs = dict(cfg)
            cfg_bfs["rw_width"] = w_bfs
            cfg_nbt = dict(cfg)
            cfg_nbt["rw_width"] = w_nbt

            X1, y1 = gen_walks(cfg_bfs, graph, "bfs")
            X2, y2 = gen_walks(cfg_nbt, graph, "nbt")

            X = torch.cat([X1, X2], dim=0)
            y = torch.cat([y1, y2], dim=0)
        else:
            X, y = gen_walks(cfg, graph, rw_mode)

        X = X.long()
        y = y.view(-1)
        y_t = y_transform_torch(y, y_mode)

        M = X.size(0)
        perm_idx = torch.randperm(M, device=device)
        X = X[perm_idx]
        y_t = y_t[perm_idx]

        # split
        try:
            bins = make_bins_np(y_t, clip=strat_clip, bin_size=strat_bin_size)
            try:
                idx = np.arange(M)
                idx_tr, idx_va = train_test_split(
                    idx, test_size=val_ratio, stratify=bins, shuffle=True, random_state=123
                )
            except ImportError:
                raise RuntimeError("sklearn not installed")
        except Exception:
            val_M = int(M * val_ratio)
            idx_va = np.arange(val_M)
            idx_tr = np.arange(val_M, M)

        idx_tr_t = torch.as_tensor(idx_tr, device=device, dtype=torch.long)
        idx_va_t = torch.as_tensor(idx_va, device=device, dtype=torch.long)
        X_tr, y_tr = X[idx_tr_t], y_t[idx_tr_t]
        X_va, y_va = X[idx_va_t], y_t[idx_va_t]

        total = 0.0
        for i in range(0, X_tr.size(0), bs):
            xb = X_tr[i:i+bs]
            yb = y_tr[i:i+bs].view(-1)
            pred = model(xb).view(-1)
            loss = loss_fn(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += float(loss.item()) * xb.size(0)
        train_loss = total / max(1, X_tr.size(0))

        model.eval()
        total = 0.0
        with torch.no_grad():
            for i in range(0, X_va.size(0), bs):
                xb = X_va[i:i+bs]
                yb = y_va[i:i+bs].view(-1)
                pred = model(xb).view(-1)
                loss = loss_fn(pred, yb)
                total += float(loss.item()) * xb.size(0)
        val_loss = total / max(1, X_va.size(0))

        sched.step()
        lr_now = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{epochs} | lr={lr_now:.2e} | train={train_loss:.5f} | val={val_loss:.5f}", flush=True)

        if es_patience and epoch >= es_warmup:
            improved = (best_val - val_loss) > es_min_delta
            if improved:
                best_val = float(val_loss)
                bad_epochs = 0
                if es_restore:
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= es_patience:
                    print(f"[early_stop] epoch={epoch} best_val={best_val:.5f} patience={es_patience} min_delta={es_min_delta}", flush=True)
                    break

    if es_patience and es_restore and best_state is not None:
        model.load_state_dict(best_state, strict=True)
