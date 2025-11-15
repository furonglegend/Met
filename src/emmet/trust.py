"""
Trust scoring utilities for EMMET edits.

MVP design:
- Uses available closed-form distance terms to derive a [0,1] trust score.
- Components:
  * gain: relative improvement (old_edit_distance - new_edit_distance) / (|old|+eps)
  * preserve: normalized preservation_distance against (|old|+eps) via tanh
- score_raw = w_gain*gain - w_preserve*preserve
- trust_score = clip((score_raw+1)/2, 0, 1)

This avoids an expensive forward pass; suitable for quick gating.
"""
from typing import Dict, Optional
import math


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def compute_trust_score(
    preservation_distance: Optional[float],
    new_edit_distance: Optional[float],
    old_edit_distance: Optional[float],
    weights: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Compute a trust score in [0,1] from distance terms.

    Returns None if inputs are insufficient.
    """
    if new_edit_distance is None or old_edit_distance is None:
        return None
    w = weights or {"gain": 0.7, "preserve": 0.3}
    eps = 1e-6

    gain_raw = max(0.0, float(old_edit_distance) - float(new_edit_distance))
    denom = abs(float(old_edit_distance)) + eps
    gain = gain_raw / denom  # in [0, +)
    # bound via tanh to [0,1)
    gain = math.tanh(gain)

    pres = max(0.0, float(preservation_distance) if preservation_distance is not None else 0.0)
    pres = math.tanh(pres / denom)

    score_raw = (w.get("gain", 0.7) * gain) - (w.get("preserve", 0.3) * pres)
    score = (score_raw + 1.0) / 2.0
    return _clip01(score)
