import importlib
import importlib.metadata

__version__ = importlib.metadata.version("mmu")
from mmu.lib import _mmu_core as core

__all__ = ["core", "__version__"]
