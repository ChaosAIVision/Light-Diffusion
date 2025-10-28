import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import yaml

# Local imports
from . import args as args_module
from .model_utils import ModelConfig


logger = logging.getLogger(__name__)


def _to_namespace(value: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace for attribute-style access."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [ _to_namespace(v) for v in value ]
    return value


def _dict_to_namespace(
    base: argparse.Namespace,
    updates: Dict[str, Any],
    *,
    strict_known: bool = False,
) -> argparse.Namespace:
    """
    Merge a dict of updates into an argparse.Namespace.

    - When strict_known is True: only existing attributes are updated (legacy behavior).
    - When strict_known is False: unknown keys are added dynamically to the namespace.
      Nested dicts are converted to SimpleNamespace for dot-access.
    """
    for k, v in updates.items():
        target_value = _to_namespace(v)
        if hasattr(base, k):
            setattr(base, k, target_value)
        else:
            if strict_known:
                logger.debug("Ignoring unknown config key: %s", k)
            else:
                setattr(base, k, target_value)
    return base


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping/dict")
    return data


def _validate_config(cfg: argparse.Namespace) -> None:
    # Keep legacy validation parity with args.py
    if cfg.resolution % 8 != 0:
        raise ValueError("resolution must be divisible by 8")


def load_training_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[argparse.Namespace, ModelConfig]:
    """
    Load training configuration from YAML and merge over existing default args.

    - When config_path is provided, fields in YAML override defaults from args.py.
    - Unknown keys at the top-level are ignored (to keep backward compatibility).
    - The nested `model:` section is used to build ModelConfig.
    - Optional cli_overrides can force-set values after YAML merge.
    """
    # Start from default CLI args for backward compatibility
    default_args = args_module.parse_args([])

    raw_cfg: Dict[str, Any] = {}
    if config_path:
        raw_cfg = _load_yaml(Path(config_path))

    # Extract and build ModelConfig from nested mapping if present
    model_section = raw_cfg.get("model", {}) if isinstance(raw_cfg, dict) else {}
    model_cfg = ModelConfig(
        pretrained_model_name_or_path=model_section.get("pretrained_model_name_or_path", getattr(default_args, "pretrained_model_name_or_path", "")),
        unet_model_name_or_path=model_section.get("unet_model_name_or_path", getattr(default_args, "unet_model_name_or_path", "")),
        vae_model_name_or_path=model_section.get("vae_model_name_or_path", getattr(default_args, "vae_model_name_or_path", "")),
        is_small_vae=bool(model_section.get("is_small_vae", getattr(default_args, "is_small_vae", False))),
    )

    # Merge remaining top-level keys into args Namespace
    top_level_updates = {k: v for k, v in raw_cfg.items() if k != "model"}
    # Permit arbitrary user-defined keys: add to args namespace (nested dicts become namespaces)
    args_ns = _dict_to_namespace(default_args, top_level_updates, strict_known=False)

    # Also attach the full YAML as args.cfg for convenience
    if raw_cfg:
        setattr(args_ns, "cfg", _to_namespace(raw_cfg))

    # Apply explicit CLI overrides if any
    if cli_overrides:
        args_ns = _dict_to_namespace(args_ns, cli_overrides)

    _validate_config(args_ns)
    return args_ns, model_cfg
