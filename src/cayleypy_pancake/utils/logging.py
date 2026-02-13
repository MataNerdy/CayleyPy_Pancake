def log_print(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)