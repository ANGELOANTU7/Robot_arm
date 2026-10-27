"""A simple parametric right hand (21 MediaPipe-ordered landmarks) for tests and synthetic demonstrations."""

import numpy as np

# Canonical right hand, palm frame: x towards the fingers, z out of the palm, y = z x x (away from the thumb, so
# the thumb and index finger are on the -y side). Metres.
_WRIST = np.array([-0.045, 0.0, 0.0])
_MCPS = np.array([[0.045, -0.025, 0.0], [0.047, -0.008, 0.0], [0.045, 0.010, 0.0], [0.040, 0.027, 0.0]])
_SEGMENTS = np.array([[0.040, 0.025, 0.020], [0.045, 0.028, 0.022], [0.042, 0.026, 0.020], [0.032, 0.020, 0.018]])


def hand_landmarks(curl: float, R: np.ndarray | None = None, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    """21 landmarks for a hand with finger curl in [0, 1] (0 open, 1 fist), palm rotation R and palm centre."""
    pts = np.zeros((21, 3))
    pts[0] = _WRIST
    bend = curl * np.pi / 2.2
    for f, (mcp, segs) in enumerate(zip(_MCPS, _SEGMENTS)):
        base = 5 + 4 * f
        pts[base] = mcp
        p, angle = mcp.copy(), 0.0
        for j, length in enumerate(segs):
            angle += bend
            p = p + length * np.array([np.cos(angle), 0.0, np.sin(angle)])
            pts[base + 1 + j] = p
    # Thumb: swings from beside the index finger towards the index knuckle as the hand closes.
    open_tip, closed_tip = np.array([0.035, -0.075, 0.025]), np.array([0.045, -0.030, 0.045])
    tip = open_tip + curl * (closed_tip - open_tip)
    cmc = np.array([-0.025, -0.022, 0.0])
    pts[1] = cmc
    pts[2] = cmc + 0.4 * (tip - cmc) + [0, -0.01, 0]
    pts[3] = cmc + 0.7 * (tip - cmc) + [0, -0.005, 0]
    pts[4] = tip
    pts -= pts[[0, 5, 9, 13, 17]].mean(axis=0)  # palm centre at the origin
    R = np.eye(3) if R is None else R
    return pts @ R.T + np.asarray(center)
