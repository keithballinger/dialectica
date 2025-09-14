from __future__ import annotations

import os
from datetime import datetime
from typing import Dict


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, no exports, no quoting."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and value and key not in os.environ:
                os.environ[key] = value


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def env_get_many(keys: Dict[str, str]) -> Dict[str, str]:
    """Return dict mapping logical name -> env value, missing become empty string."""
    out: Dict[str, str] = {}
    for logical, env_key in keys.items():
        out[logical] = os.environ.get(env_key, "")
    return out


def is_dry_run() -> bool:
    return os.environ.get("DIALECTICA_DRY_RUN", "0") in {"1", "true", "yes", "on"}

