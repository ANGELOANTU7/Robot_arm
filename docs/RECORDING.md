# Recording your demonstrations

The robot learns from short phone videos of you picking up a ball, moving it over a cup and dropping it in.
A single fixed camera is enough. A printed marker sheet gives the video a metric table frame, and the pipeline
recovers the 3D path of your thumb–index pinch from the hand landmarks.

## What you need

| Item | Notes |
|---|---|
| Phone + tripod (or a stack of books) | The camera must not move during a clip. |
| `docs/marker_sheet.pdf` printed on A4 | Print at **100% scale** (no "fit to page"). Measure a black square: it should be 50 mm. |
| A small, brightly colored ball | A 4 cm orange ping-pong ball is ideal (the default `--ball-color orange`). |
| A cup of one solid color | Blue is the default (`--cup-color blue`). About 8 cm wide; measure the base diameter. |
| A plain table | It must not be orange or blue. Avoid orange or blue objects in the frame. |

## Setup (seen from above)

```
                you (reach in from this side)
        ┌─────────────────────────────────────┐
        │        ●ball              ▢cup      │   ← ball and cup: anywhere here, off the sheet
        │                                     │
        │         ┌───────────────────┐       │
        │         │  ▣             ▣  │       │   ← marker sheet, flat, on the camera's side
        │         │        y↑ →x      │       │     of the objects so nothing covers it
        │         │  ▣             ▣  │       │
        │         └───────────────────┘       │
        └─────────────────────────────────────┘
                         📱 camera
                 50–80 cm away, 30–50° above the table
```

- Use landscape orientation, 1080p, 30 or 60 fps, and the normal 1× lens (not ultra-wide; don't zoom).
- Frame the shot so that the whole sheet, the ball, the cup and your hand's whole path are in view.
- Lock focus and exposure if your phone allows it (on iPhone, tap and hold).
- Use even lighting and avoid strong shadows on the ball.

## Each clip (about 5–8 seconds)

1. **Hold still for 1 second** with your hand out of the frame or well away. The sheet, ball and cup must be
   fully visible: the first second is used for calibration.
2. Reach for the ball and **pinch it between your thumb and index finger**. Keep the ball visible between your
   fingertips; don't hide it in your fist.
3. Lift it about 10–15 cm above the cup rim and move it over the cup.
4. Pause briefly over the cup, then **open your thumb and index finger** to drop the ball in.
5. Withdraw your hand and **hold still for another second**.

Use one hand only, and keep any second hand out of the frame.

## Variety

Record **10–20 clips**. Between clips, move the ball and the cup to new spots (15–40 cm apart, never on the
sheet) and vary your path a little: higher, lower, a curve. The robot sees your motion relative to the ball and the cup,
so variety in the layout teaches it to generalise.

## Processing

```bash
mkdir -p data/videos && cp /path/to/your/clips/*.mp4 data/videos/

# 1. Quick check on one clip: are all 4 markers found, and is only the ball orange / the cup blue?
python scripts/process_video.py data/videos/clip01.mp4 --check

# 2. Process all clips (use your measured sizes)
python scripts/process_video.py data/videos/*.mp4 --ball-diameter-cm 4 --cup-diameter-cm 8
```

For each clip, open `data/processed/human/<clip>_overlay.mp4` and `<clip>_trajectory.png`:

- The yellow path should follow your fingertips, and **HOLD** should cover the time you held the ball.
- `scale` in the printed summary is the ratio between your hand size and MediaPipe's average hand. It should
  be within about 0.8–1.25.
- `release_offset_m` is how far the reconstructed hand was from the cup at release, before correction. A few
  centimetres is normal.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Could not find at least 2 ArUco markers" | Make the sheet larger in the image (move the camera closer), improve the lighting, or keep it flat. |
| Ball or cup not detected | Run `--check` and look at the highlighted pixels. Pass a custom range with `--ball-hsv H,S,V,H,S,V` (OpenCV HSV; H is 0–179). |
| `scale` outside 0.7–1.4 | The focal length estimate is off. Pass `--hfov-deg` (most phone main cameras record video at about 65–75°) or `--focal-px`. |
| Grasp or release at the wrong time | Pass `--grasp-time` and `--release-time` (seconds from when your hand first appears; the overlay prints `t`). |
| Printed markers aren't 50 mm | Pass the measured size with `--marker-mm`. |
