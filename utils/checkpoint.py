import warnings
from pathlib import Path
from typing import Optional

import torch


def _adapt_hash_heads_for_model(checkpoint_model_state: dict, model) -> dict:
    model_state = model.state_dict()
    adapted = dict(checkpoint_model_state)

    full_weight = adapted.get("hash_head.fc.weight", None)
    full_bias = adapted.get("hash_head.fc.bias", None)
    sem_weight_key = "semantic_hash_head.fc.weight"
    sem_bias_key = "semantic_hash_head.fc.bias"
    tmp_weight_key = "temporal_hash_head.fc.weight"
    tmp_bias_key = "temporal_hash_head.fc.bias"

    if full_weight is not None and sem_weight_key in model_state and tmp_weight_key in model_state:
        sem_rows = model_state[sem_weight_key].shape[0]
        tmp_rows = model_state[tmp_weight_key].shape[0]
        needed_rows = sem_rows + tmp_rows
        if full_weight.ndim == 2 and full_weight.shape[0] >= needed_rows:
            if full_weight.shape[1:] == model_state[sem_weight_key].shape[1:]:
                adapted.setdefault(sem_weight_key, full_weight[:sem_rows])
                adapted.setdefault(tmp_weight_key, full_weight[sem_rows:needed_rows])
        if full_bias is not None and full_bias.ndim == 1 and full_bias.shape[0] >= needed_rows:
            adapted.setdefault(sem_bias_key, full_bias[:sem_rows])
            adapted.setdefault(tmp_bias_key, full_bias[sem_rows:needed_rows])

    if "hash_head.fc.weight" in model_state and sem_weight_key in adapted and tmp_weight_key in adapted:
        sem_weight = adapted[sem_weight_key]
        tmp_weight = adapted[tmp_weight_key]
        merged_weight = torch.cat([sem_weight, tmp_weight], dim=0)
        if merged_weight.shape == model_state["hash_head.fc.weight"].shape:
            adapted.setdefault("hash_head.fc.weight", merged_weight)
        sem_bias = adapted.get(sem_bias_key, None)
        tmp_bias = adapted.get(tmp_bias_key, None)
        if sem_bias is not None and tmp_bias is not None:
            merged_bias = torch.cat([sem_bias, tmp_bias], dim=0)
            if merged_bias.shape == model_state["hash_head.fc.bias"].shape:
                adapted.setdefault("hash_head.fc.bias", merged_bias)

    return adapted


def _load_model_state_compatible(model, checkpoint_model_state: dict):
    adapted = _adapt_hash_heads_for_model(checkpoint_model_state, model)
    model_state = model.state_dict()
    unexpected_adapted = [
        key
        for key in adapted
        if key not in model_state and not key.startswith("hash_head.")
    ]
    loadable = {key: value for key, value in adapted.items() if key in model_state}
    incompatible = model.load_state_dict(loadable, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("hash_head.")
    ]
    unexpected = unexpected_adapted + [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith("hash_head.")
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Error(s) in loading state_dict for "
            f"{model.__class__.__name__}: missing_keys={missing}, unexpected_keys={unexpected}"
        )
    return incompatible


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
    _load_model_state_compatible(model, state["model"])
    if optimizer is not None and "optimizer" in state:
        try:
            optimizer.load_state_dict(state["optimizer"])
            state["_optimizer_loaded"] = True
        except (RuntimeError, ValueError) as exc:
            state["_optimizer_loaded"] = False
            state["_optimizer_load_error"] = str(exc)
            warnings.warn(
                "Skipped optimizer state while loading checkpoint because it is incompatible "
                "with the current model parameters. Model weights were loaded.",
                RuntimeWarning,
                stacklevel=2,
            )
    if scheduler is not None and "scheduler" in state:
        if optimizer is not None and "optimizer" in state and not state.get("_optimizer_loaded", False):
            state["_scheduler_loaded"] = False
            state["_scheduler_load_error"] = "optimizer state was skipped"
            warnings.warn(
                "Skipped scheduler state because optimizer state was not restored.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            try:
                scheduler.load_state_dict(state["scheduler"])
                state["_scheduler_loaded"] = True
            except (RuntimeError, ValueError) as exc:
                state["_scheduler_loaded"] = False
                state["_scheduler_load_error"] = str(exc)
                warnings.warn(
                    "Skipped scheduler state while loading checkpoint because it is incompatible "
                    "with the current optimizer.",
                    RuntimeWarning,
                    stacklevel=2,
                )
    if criterion is not None and "criterion" in state:
        criterion.load_state_dict(state["criterion"], strict=False)
    return state
