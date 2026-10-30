import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
DATA = os.path.join(ROOT, "data", "f1_raw.csv")

# Make sure THIS project's src/ wins over any other folder on the path,
# and drop a same-named module that may already have been imported.
if SRC in sys.path:
    sys.path.remove(SRC)
sys.path.insert(0, SRC)
for name in ("features", "validate", "train", "metrics"):
    mod = sys.modules.get(name)
    if mod is not None and not getattr(mod, "__file__", "").startswith(SRC):
        del sys.modules[name]