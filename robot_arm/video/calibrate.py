"""Camera calibration from a printed ArUco sheet lying flat on the table.

Table frame: origin at the centre of the sheet, x along the long edge (towards marker 1), y along the short edge
(towards marker 3), z up out of the table. Units are metres.
"""

from dataclasses import dataclass

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50
SHEET_MM = (297.0, 210.0)  # A4 landscape
MARKER_MM = 50.0
MARGIN_MM = 15.0
MARKER_IDS = (0, 1, 2, 3)  # bottom-left, bottom-right, top-left, top-right (seen from above)


def marker_centers_m(sheet_mm=SHEET_MM, marker_mm=MARKER_MM, margin_mm=MARGIN_MM) -> dict[int, np.ndarray]:
    off_x = (sheet_mm[0] / 2 - margin_mm - marker_mm / 2) / 1000
    off_y = (sheet_mm[1] / 2 - margin_mm - marker_mm / 2) / 1000
    return {0: np.array([-off_x, -off_y]), 1: np.array([off_x, -off_y]),
            2: np.array([-off_x, off_y]), 3: np.array([off_x, off_y])}


def marker_corners_table(marker_mm: float = MARKER_MM) -> dict[int, np.ndarray]:
    """3D corners of each marker in the table frame, in ArUco order (TL, TR, BR, BL as printed)."""
    h = marker_mm / 2000
    corners = {}
    for mid, c in marker_centers_m(marker_mm=marker_mm).items():
        # Printed image: +x right, +y up on the table; ArUco's "top-left" is the printed top-left.
        corners[mid] = np.array([[c[0] - h, c[1] + h, 0], [c[0] + h, c[1] + h, 0],
                                 [c[0] + h, c[1] - h, 0], [c[0] - h, c[1] - h, 0]])
    return corners


def make_marker_sheet(dpi: int = 300, marker_mm: float = MARKER_MM) -> np.ndarray:
    """Printable A4-landscape image with the four markers and a centre cross."""
    px = lambda mm: int(round(mm / 25.4 * dpi))  # noqa: E731
    w, h = px(SHEET_MM[0]), px(SHEET_MM[1])
    sheet = np.full((h, w), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    size = px(marker_mm)
    for mid, c in marker_centers_m(marker_mm=marker_mm).items():
        img = cv2.aruco.generateImageMarker(dictionary, mid, size)
        cx = px(c[0] * 1000 + SHEET_MM[0] / 2)
        cy = px(SHEET_MM[1] / 2 - c[1] * 1000)  # image rows grow downwards
        sheet[cy - size // 2:cy - size // 2 + size, cx - size // 2:cx - size // 2 + size] = img
    # Table-frame axes at the sheet centre.
    o = (w // 2, h // 2)
    thick = max(2, dpi // 60)
    cv2.arrowedLine(sheet, o, (o[0] + px(25), o[1]), 0, thick, tipLength=0.25)
    cv2.arrowedLine(sheet, o, (o[0], o[1] - px(25)), 0, thick, tipLength=0.25)
    cv2.putText(sheet, "x", (o[0] + px(27), o[1] + px(2)), cv2.FONT_HERSHEY_SIMPLEX, dpi / 100, 0, thick)
    cv2.putText(sheet, "y", (o[0] - px(2), o[1] - px(28)), cv2.FONT_HERSHEY_SIMPLEX, dpi / 100, 0, thick)
    cv2.putText(sheet, f"print at 100% - marker side must measure {marker_mm:.0f} mm",
                (px(MARGIN_MM), h - px(4)), cv2.FONT_HERSHEY_SIMPLEX, dpi / 300, 0, 2)
    return sheet


def detect_markers(frame_bgr: np.ndarray) -> dict[int, np.ndarray]:
    """Marker id -> (4, 2) pixel corners for the sheet's markers found in the frame."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
                                       cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners) if int(i) in MARKER_IDS}


@dataclass
class Camera:
    """Pinhole camera with pose: p_cam = R @ p_table + t."""

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int
    height: int

    @property
    def center_table(self) -> np.ndarray:
        """Camera optical centre in the table frame."""
        return -self.R.T @ self.t

    def to_cam(self, p_table: np.ndarray) -> np.ndarray:
        return p_table @ self.R.T + self.t

    def to_table(self, p_cam: np.ndarray) -> np.ndarray:
        return (p_cam - self.t) @ self.R

    def project(self, p_table: np.ndarray) -> np.ndarray:
        pc = self.to_cam(np.atleast_2d(p_table))
        uv = pc @ self.K.T
        return uv[:, :2] / uv[:, 2:3]

    def ray(self, uv) -> np.ndarray:
        """Unit viewing direction through pixel uv, in the table frame."""
        d_cam = np.linalg.solve(self.K, np.array([uv[0], uv[1], 1.0]))
        d = self.R.T @ d_cam
        return d / np.linalg.norm(d)

    def backproject_to_plane(self, uv, z: float = 0.0) -> np.ndarray:
        """Point where the ray through pixel uv hits the horizontal plane at height z (table frame)."""
        o, d = self.center_table, self.ray(uv)
        s = (z - o[2]) / d[2]
        return o + s * d

    def as_dict(self) -> dict:
        return {"K": self.K, "R": self.R, "t": self.t, "size": np.array([self.width, self.height])}

    @classmethod
    def from_dict(cls, d) -> "Camera":
        w, h = (int(v) for v in d["size"])
        return cls(np.asarray(d["K"]), np.asarray(d["R"]), np.asarray(d["t"]), w, h)


def intrinsics_from_fov(width: int, height: int, hfov_deg: float = 68.0) -> np.ndarray:
    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def calibrate(frames_bgr: list[np.ndarray], marker_mm: float = MARKER_MM, focal_px: float | None = None,
              hfov_deg: float | None = None) -> tuple[Camera, dict]:
    """Estimate intrinsics (focal length only) and the table pose from frames showing the marker sheet.

    The focal length comes from (in priority order): `focal_px`, `hfov_deg`, or a single-view calibration on the
    sheet's 16 corners (needs the camera to look at the sheet at an angle, not straight down).
    """
    h, w = frames_bgr[0].shape[:2]
    table_corners = marker_corners_table(marker_mm)
    obj, img = None, None
    for frame in frames_bgr:  # use the frame where the most markers are visible
        found = detect_markers(frame)
        if len(found) >= 2 and (obj is None or len(found) * 4 > len(obj)):
            obj = np.concatenate([table_corners[i] for i in sorted(found)]).astype(np.float32)
            img = np.concatenate([found[i] for i in sorted(found)]).astype(np.float32)
    if obj is None:
        raise RuntimeError("Could not find at least 2 ArUco markers of the sheet in the first frames. "
                           "Check lighting and that the whole sheet is visible (see docs/RECORDING.md).")

    info = {"markers_found": len(obj) // 4}
    if focal_px:
        K, info["focal_source"] = np.array([[focal_px, 0, w / 2], [0, focal_px, h / 2], [0, 0, 1.0]]), "given"
    elif hfov_deg:
        K, info["focal_source"] = intrinsics_from_fov(w, h, hfov_deg), "fov"
    else:
        K = intrinsics_from_fov(w, h)
        info["focal_source"] = "fov-default"
        if len(obj) >= 12:
            flags = (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
                     | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_USE_INTRINSIC_GUESS)
            rms, K_est, _, _, _ = cv2.calibrateCamera([obj], [img], (w, h), K.copy(), np.zeros(5), flags=flags)
            f = K_est[0, 0]
            if rms < 2.0 and 0.4 * w < f < 3.0 * w:
                K, info["focal_source"], info["calib_rms_px"] = K_est, "aruco", float(rms)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("solvePnP failed on the marker sheet.")
    R, _ = cv2.Rodrigues(rvec)
    cam = Camera(K, R, tvec.ravel(), w, h)
    info["reproj_err_px"] = float(np.linalg.norm(cam.project(obj.astype(float)) - img, axis=1).mean())
    info["focal_px"] = float(K[0, 0])
    info["camera_height_m"] = float(cam.center_table[2])
    return cam, info
