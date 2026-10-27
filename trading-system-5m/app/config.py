from pathlib import Path
import yaml

BASE_DIR = Path("/root/trading-system")
CONFIG_DIR = BASE_DIR / "config"

def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        return {}
    return data

def load_config() -> dict:
    settings = _read_yaml(CONFIG_DIR / "settings.yaml")
    pairs = _read_yaml(CONFIG_DIR / "pairs.yaml")
    risk = _read_yaml(CONFIG_DIR / "risk.yaml")
    strategies = _read_yaml(CONFIG_DIR / "strategies.yaml")

    return {
        "settings": settings,
        "pairs": pairs.get("pairs", []),
        "risk": risk.get("risk", {}),
        "strategies": strategies.get("strategies", {}),
    }
