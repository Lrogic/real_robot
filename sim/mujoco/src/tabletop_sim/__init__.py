"""Config-driven mjlab tabletop manipulation environments."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PACKAGE_ROOT / "assets"
CONFIGS_DIR = PACKAGE_ROOT / "configs"
