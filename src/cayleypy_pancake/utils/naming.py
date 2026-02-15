def exp_name_from(cfg: dict, *, n: int, w_gap: float, gap_mode: str) -> str:
    mt = cfg.get("model_type", "EmbMLP")

    if mt == "EmbMLP":
        ed  = cfg.get("emb_dim", 0)
        pos = int(bool(cfg.get("use_pos_emb", True)))
        nrd = int(cfg.get("nrd", 0))
        tag = f"{mt}_ed{ed}_pos{pos}_nrd{nrd}"
    else:
        nrd = int(cfg.get("nrd", 0))
        tag = f"{mt}_nrd{nrd}"

    rw = int(cfg.get("rw_width", 0))
    mf = cfg.get("mix_bfs_frac", None)
    mf_tag = f"_mix{float(mf):.2f}" if mf is not None else ""

    lr = cfg.get("lr", None)
    wd = cfg.get("weight_decay", None)
    ep = int(cfg.get("num_epochs", 0))

    return (
        f"n{n}_{tag}"
        f"_rw{rw}{mf_tag}"
        f"_lr{lr}_wd{wd}_ep{ep}"
        f"_wg{w_gap}_{gap_mode}"
    )
