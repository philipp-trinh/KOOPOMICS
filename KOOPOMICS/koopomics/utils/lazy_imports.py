# koopomics/utils/lazy_imports.py

import importlib

class LazyImport:
    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return self._module

    def __getattr__(self, name):
        return getattr(self._load(), name)

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __repr__(self):
        return f"<LazyImport {self._module_name}>"


def make_lazy_module(import_map):
    def __getattr__(name):
        if name in import_map:
            module = importlib.import_module(import_map[name])
            return getattr(module, name)
        raise AttributeError(f"module has no attribute {name}")
    return __getattr__
