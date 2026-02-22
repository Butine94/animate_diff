"""
Prompt loading utilities.

Supports three input formats:
  - YAML file with a top-level ``shots`` list
  - Plain-text file (one prompt per line; ``#`` comments ignored)
  - Config-embedded ``fallback_prompts`` list

Each returned shot dict has ``id``, ``prompt``, and ``method`` keys.
``method`` is ``"animatediff"`` (default) or ``"sd_still"``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_VALID_METHODS = {"animatediff", "sd_still"}


def load_prompts(path: str, max_shots: int | None = None) -> list[dict]:
    """
    Load shots from a YAML or plain-text prompt file.

    Parameters
    ----------
    path:
        Path to a ``.yaml`` / ``.yml`` or plain-text file.
    max_shots:
        Optional cap on returned shots.
    """
    filepath = Path(path)
    if not filepath.exists():
        logger.error("Prompt file not found: %s", path)
        return []

    shots = (
        _load_yaml(filepath)
        if filepath.suffix.lower() in {".yaml", ".yml"}
        else _load_text(filepath)
    )

    if max_shots is not None:
        shots = shots[:max_shots]

    logger.info("Loaded %d shot(s) from %s.", len(shots), path)
    return shots


def prompts_from_config(config: dict, max_shots: int = 5) -> list[dict]:
    """
    Build a shot list from ``config['generation']['fallback_prompts']``.
    Returns ``[]`` if no fallbacks are defined.
    """
    fallbacks: list[str] = config.get("generation", {}).get("fallback_prompts", [])
    if not fallbacks:
        logger.warning("No fallback_prompts in config.")
        return []
    return _shots_from_strings(fallbacks[:max_shots])


# ------------------------------------------------------------------
# Parsers
# ------------------------------------------------------------------

def _load_yaml(filepath: Path) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "shots" not in data:
        logger.error(
            "YAML file must have a top-level 'shots' list. Got: %s",
            type(data).__name__,
        )
        return []

    shots: list[dict] = []
    for i, entry in enumerate(data["shots"]):
        if isinstance(entry, str):
            shots.append(_make_shot(i, entry))
        elif isinstance(entry, dict):
            prompt = entry.get("prompt", "").strip()
            if not prompt:
                logger.warning("Shot %d missing 'prompt' — skipping.", i + 1)
                continue
            method = entry.get("method", "animatediff")
            if method not in _VALID_METHODS:
                logger.warning(
                    "Shot %d has unknown method '%s' — defaulting to animatediff.", i + 1, method
                )
                method = "animatediff"
            shots.append({"id": i + 1, "prompt": prompt, "method": method})
        else:
            logger.warning("Unrecognised entry type at index %d: %s", i, type(entry))

    return shots


def _load_text(filepath: Path) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
    prompts = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    return _shots_from_strings(prompts)


def _shots_from_strings(prompts: list[str]) -> list[dict]:
    return [_make_shot(i, p) for i, p in enumerate(prompts)]


def _make_shot(index: int, prompt: str) -> dict:
    return {"id": index + 1, "prompt": prompt, "method": "animatediff"}