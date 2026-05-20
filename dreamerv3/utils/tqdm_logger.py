# dreamerv3/utils/tqdm_output.py
from typing import Any, Dict, Iterable, Tuple
from tqdm.auto import tqdm

def _to_dict(obj) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        try:
            return dict(obj)  # elements.Logger passes tuple of (key, value) items
        except Exception:
            return {}
    return {}

class TqdmOutput:
    """elements.Logger-compatible output. Called with (key, value) pairs tuple."""
    def __init__(self, total_steps: int, desc: str = "training"):
        self.total = int(total_steps)
        self.pbar = tqdm(total=self.total, desc=desc, unit="step", dynamic_ncols=True)
        self._last = 0

    def __call__(self, metrics: Iterable[Tuple[str, Any]] | Dict[str, Any]):
        m = _to_dict(metrics)
        if not m:
            return
        s = m.get("step")
        if isinstance(s, (int, float)):
            s = int(s)
            if s > self._last:
                self.pbar.update(s - self._last)
                self._last = s
                if self._last >= self.total:
                    self.pbar.close()

    def close(self):
        try:
            self.pbar.close()
        except Exception:
            pass
