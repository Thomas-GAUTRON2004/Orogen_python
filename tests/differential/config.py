"""Load the declarative differential registry from config.json."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .adapters import PAIRS as _ADAPTER_PAIRS

_CONFIG_PATH = Path(__file__).with_name("config.json")
_CONFIG = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))

if _CONFIG.get("version") != 1:
    raise ValueError(f"Unsupported differential config version: {_CONFIG.get('version')!r}")

_ADAPTERS = {pair.javascript_id: pair for pair in _ADAPTER_PAIRS}
_CONFIGURED_IDS = [entry["javascript_id"] for entry in _CONFIG["pairs"]]

if len(_CONFIGURED_IDS) != len(set(_CONFIGURED_IDS)):
    raise ValueError("Duplicate javascript_id in differential config.json")

_missing = set(_CONFIGURED_IDS) - _ADAPTERS.keys()
_unconfigured = _ADAPTERS.keys() - set(_CONFIGURED_IDS)
if _missing or _unconfigured:
    raise ValueError(
        "Differential registry mismatch: "
        f"missing adapters={sorted(_missing)}, "
        f"unconfigured adapters={sorted(_unconfigured)}"
    )

_pairs = []
for entry in _CONFIG["pairs"]:
    adapter = _ADAPTERS[entry["javascript_id"]]
    if adapter.python_call.__name__ != entry["python_adapter"]:
        raise ValueError(
            f"Python adapter mismatch for {entry['javascript_id']!r}: "
            f"expected {entry['python_adapter']!r}, "
            f"found {adapter.python_call.__name__!r}"
        )
    case_ids = [case.id for case in adapter.cases]
    if case_ids != entry["cases"]:
        raise ValueError(
            f"Case IDs mismatch for {entry['javascript_id']!r}: "
            f"JSON={entry['cases']!r}, Python={case_ids!r}"
        )
    _pairs.append(
        replace(
            adapter,
            name=entry["name"],
            rtol=entry["rtol"],
            atol=entry["atol"],
        )
    )

PAIRS = tuple(_pairs)
