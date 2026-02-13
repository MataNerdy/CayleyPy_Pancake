def exp_name_from(cfg: dict, *, n: int, w_gap: float, gap_mode: str):
    mt = cfg.get("model_type", "EmbMLP")
    nrd = int(cfg.get("nrd", 0))
    if mt == "EmbMLP":
        ed = int(cfg.get("emb_dim", -1))
        pos = 1 if bool(cfg.get("use_pos_emb", True)) else 0
        return f"n{n}_{mt}_ed{ed}_pos{pos}_nrd{nrd}_wg{w_gap}_{gap_mode}"
    if mt == "MLPRes1":
        return f"n{n}_{mt}_nrd{nrd}_wg{w_gap}_{gap_mode}"
    return f"n{n}_{mt}_wg{w_gap}_{gap_mode}"
