import pkgutil
import importlib

def load_plugins() -> None:
    """Import all modules under mas_finance.inventories so their @register(...) decorators run."""
    pkg = __name__
    for finder, name, ispkg in list(pkgutil.iter_modules(__path__)):
        if name.startswith("_"):
            continue
        importlib.import_module(f"{pkg}.{name}")
