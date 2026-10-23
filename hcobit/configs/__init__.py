import importlib.util


def load_config(spec):
    """Load "path/to/config.py[:arg]" -> get_config(arg)."""
    path, _, arg = spec.partition(":")
    mod_spec = importlib.util.spec_from_file_location("hcobit_config", path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return mod.get_config(arg) if arg else mod.get_config()
