"""Legacy import compatibility. Active experiment parameters live in main.py.

No filesystem scans or import-time printing. Existing external callers can still
import HybridSOSConfig; unused legacy update fields remain on that dataclass.
"""
from .env import HybridSOSConfig
