"""
Inventory module - contains all pluggable methods for agents.

Each agent type has a sub-module with its available methods:
- analyst/: Feature construction and trend detection methods
- researcher/: Forecasting, uncertainty, and calibration methods
- trader/: Execution style methods
- risk/: Risk check methods
"""
import pkgutil
import importlib
from pathlib import Path

def load_all_methods() -> None:
    """Import all inventory modules so their @register decorators run."""
    pkg_path = Path(__file__).parent

    # Load top-level modules
    for finder, name, ispkg in pkgutil.iter_modules([str(pkg_path)]):
        if name.startswith("_"):
            continue
        if ispkg:
            # Load sub-package modules
            subpkg_path = pkg_path / name
            for _, subname, _ in pkgutil.iter_modules([str(subpkg_path)]):
                if not subname.startswith("_"):
                    importlib.import_module(f".{name}.{subname}", package=__name__)
        else:
            importlib.import_module(f".{name}", package=__name__)

from .registry import register, get, list_methods

__all__ = ["register", "get", "list_methods", "load_all_methods"]
