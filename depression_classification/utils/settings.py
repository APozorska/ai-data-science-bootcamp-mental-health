import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CFG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"

load_dotenv()
CFG_PATH: Path = Path(os.getenv("CFG_PATH") or DEFAULT_CFG_PATH)
