"""Central configuration loaded from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny stdlib-only .env loader."""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT_DIR / ".env")

# Gemini configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Project paths
KNOWLEDGE_BASE_DIR = (
    ROOT_DIR / "knowledge_base_override"
    if (ROOT_DIR / "knowledge_base_override").exists()
    else ROOT_DIR / "knowledge-base"
)

ORDERS_PATH = ROOT_DIR / "data" / "orders.json"

# Retrieval tuning
TOP_K_CHUNKS = int(os.environ.get("TOP_K_CHUNKS", "5"))

RETRIEVAL_HISTORY_TURNS = int(
    os.environ.get("RETRIEVAL_HISTORY_TURNS", "2")
)

LOW_CONFIDENCE_FLOOR = float(
    os.environ.get("LOW_CONFIDENCE_FLOOR", "6.0")
)

# System prompt
SYSTEM_PROMPT_VARIANT = os.environ.get(
    "SYSTEM_PROMPT_VARIANT",
    "final",
)

# Tool loop
MAX_TOOL_ITERATIONS = 4