"""Make the printable ArUco sheet that defines the table frame in your videos.

    python scripts/make_markers.py      # writes docs/marker_sheet.pdf and .png (A4 landscape)

Print the PDF at 100% scale (no "fit to page"), then check that each black square measures 50 mm. If it doesn't,
pass the measured size to process_video.py with --marker-mm.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from robot_arm.video.calibrate import SHEET_MM, make_marker_sheet  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="docs/marker_sheet", help="output path without extension")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet = make_marker_sheet(dpi=args.dpi)
    cv2.imwrite(str(out.with_suffix(".png")), sheet, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    # The PDF page is exactly A4, so printing at 100% gives true-size markers.
    fig = plt.figure(figsize=(SHEET_MM[0] / 25.4, SHEET_MM[1] / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(sheet, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.axis("off")
    fig.savefig(out.with_suffix(".pdf"), dpi=args.dpi)
    print(f"Wrote {out.with_suffix('.pdf')} and .png: print the PDF on A4 at 100% scale.")


if __name__ == "__main__":
    main()
