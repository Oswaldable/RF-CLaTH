import ast
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    yaml = None
    _YAML_IMPORT_ERROR = exc


def _parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    if raw == "":
        return ""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.lower() in {"null", "none"}:
        return None
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    if raw.startswith("[") and raw.endswith("]"):
        return ast.literal_eval(raw)
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _load_simple_yaml(path: str) -> Dict[str, Any]:
    """Small YAML subset loader for this repo's default config.

    It supports nested mappings via two-space indentation and inline lists.
    PyYAML is still preferred when installed.
    """
    root: Dict[str, Any] = {}
    stack = [(-1, root)]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            key, sep, raw_value = line.strip().partition(":")
            if not sep:
                continue
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if raw_value.strip() == "":
                node: Dict[str, Any] = {}
                parent[key] = node
                stack.append((indent, node))
            else:
                parent[key] = _parse_scalar(raw_value)
    return root


def load_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        return _load_simple_yaml(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: Dict[str, Any], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if yaml is None:
            json.dump(cfg, f, indent=2)
        else:
            yaml.safe_dump(cfg, f, sort_keys=False)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def set_by_dotted_key(cfg: Dict[str, Any], dotted_key: str, value: Any):
    target = cfg
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    result = deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw) if yaml is not None else _parse_scalar(raw)
        set_by_dotted_key(result, key, value)
    return result
