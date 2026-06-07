from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    criterion=None,
    epoch: int = 0,
    best_metric: float = 0.0,
    cfg: dict = None,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "cfg": cfg,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if criterion is not None:
        state["criterion"] = criterion.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    criterion=None,
    map_location: Optional[str] = None,
):
    state = torch.load(path, map_location=map_location or "cpu")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if criterion is not None and "criterion" in state:
        criterion.load_state_dict(state["criterion"], strict=False)
    return state
