# Recording the cup-pyramid skills

The G1 learns the pyramid task from short phone clips of **one skill each**:

| Clip (folder or name) | What you do | How many |
|---|---|---|
| `part_a/` or `place_base_NN.mp4` | Take a mouth-down cup from wherever it lies and set it down mouth-down at its pyramid spot; then the second cup next to it (one clip may show both, A then B). | ~10 |
| `part_b/` or `place_top_NN.mp4` | Take the third cup (lying mouth-down like the others), turn it over in the air with one hand and set it mouth-up across the two base cups. | ~10 |
| `part_c/` or `toss_NN.mp4` | Pinch the bottle cap off the table and throw it into the top cup. After a miss, pick it up again and throw again until it goes in; then a thumbs-up. | ~10 |

One fixed camera is enough. The printed marker sheet gives each clip a metric table frame. The pipeline
recovers your **palm's 3D position and orientation**, how closed your hand is, the moment you grasp and release,
where the cup started and ended, and, for throws, the cap's flight and where you released it.

## What you need

| Item | Notes |
|---|---|
| Phone + tripod (or a stack of books) | The camera must not move during a clip. Hand-held clips can't be used. |
| `docs/marker_sheet.pdf` printed on A4 | Print at **100% scale**. Measure a black square: it should be 50 mm (else pass `--marker-mm`). |
| 3 red party cups, all the same | The defaults match yours: 12 cm tall, 9 cm across the mouth, 5 cm across the bottom, white rolled rim (`--cup-height-cm`, `--cup-mouth-cm`, `--cup-base-cm`, `--lip-mm`). |
| The blue bottle cap | 2.8 cm across, 1.2 cm tall (`--cap-diameter-cm`, `--cap-height-cm`). |
| A plain table | Your wooden table is fine. **Keep every other red or blue thing out of view**, e.g. the blue pen in your setup video. |

Use your **right hand** only. The robot uses its right hand, and the palm orientation assumes a right hand.

## Setup (seen from above)

```
                         you (reach in from this side)
        ┌─────────────────────────────────────────────────────┐
        │  supply area           cap                          │
        │  (your right):         ●       pyramid spot:        │
        │   ▽top                          B ▲ ▲ A             │  ← your right is on the left of this picture
        │   ▲A                              (top ▽ on B+A)    │
        │      ▲B                                             │
        │              ┌───────────────────┐                  │
        │              │  ▣             ▣  │                  │  ← marker sheet, flat, between the camera
        │              │        y↑ →x      │                  │    and the pyramid
        │              │  ▣             ▣  │                  │
        │              └───────────────────┘                  │
        └─────────────────────────────────────────────────────┘
                                 📱 camera
                   80–100 cm away, 30–45° above the table
```

(▲ = mouth down, ▽ = mouth up.)

- The two base cups stand **side by side across the camera's view** (left and right in the image, as in your
  setup video), not one behind the other. Then both are seen separately, and a throw along them crosses the image.
- Keep the supply cups **off to the side**, so that in the image a supply cup's spot doesn't overlap the pyramid.
- The camera must see the whole sheet, all cups, the cap and **your whole hand path**, including the hand lifted
  above the cups and the cap's whole flight.
- Use landscape, 1080p and the normal 1× lens. Record at **60 fps**. It is needed for the throws and fine for the
  rest.
- Use bright, even light. Lock focus and exposure if your phone allows it.
- **Copy the original files** off the phone (cable, AirDrop, Google Drive or Photos at original quality). Messaging
  apps like WhatsApp shrink a clip to about 830×460 at 30 fps, which is too coarse for the cap's flight.

## Each clip (about 4–8 seconds)

Always start and end by **holding still for 1 second** with your hand **out of the frame**. The cups all look
alike, so the pipeline finds the cup you moved by comparing the first and last second: what is red only at the
start is where the cup was, and what is red only at the end is where it went. The first second also calibrates
the camera from the sheet.

### place_base

1. Grab the cup **from the side with your whole hand**, like holding a glass: thumb up, palm against the cup,
   about halfway up. Don't grab it from above or by the rim.
2. Lift it about 10 cm, carry it over its spot and lower it until it stands on the table, mouth down.
3. Open your hand and pull it back out of the frame.

Record cup A alone at the pyramid spot and cup B being placed **next to A, rims almost touching**, either as
separate clips or both in one clip (A first). Move the cups around between clips: they can lie anywhere.

### place_top

1. Start with A and B standing side by side, mouth down. The top cup lies mouth down somewhere on the table.
2. Grab it from the side, lift it, **turn it over in the air** with the same hand and set it down **centred
   across both base cups**, mouth up.
3. Open your hand gently and pull it back out of the frame. The pyramid must stay standing.

### toss

1. Start with the pyramid built and the cap lying flat on the table, to your right of the pyramid.
2. **Pinch** the cap between your thumb and index finger and lift it.
3. Move your hand to the side of the pyramid, **in line with the two base cups** (on your right, cup B's side),
   **5–30 cm** from the top cup and a little above its rim.
4. Throw it **sideways along that line** with a short flick so it drops into the top cup. Thrown along the base
   cups, the cap's push on landing goes the way the top cup is held by both of them; thrown from the front, it
   would tip the top cup over.

5. If it misses, pick the cap up where it landed and throw again, until it is in. Then show a **thumbs-up**
   for a moment, and take your hand out of the frame.

Each throw is found where the cap leaves your fingers; the last one is the one that went in. The robot copies your
throw's style (release distance, height, how far off the line, arc and wind-up). Its aim is corrected in
simulation. The robot's own throw works best from about 8–20 cm, so throws from further away are pulled into
that range.

## Processing

```bash
mkdir -p data/videos_pyramid && cp /path/to/clips/*.mp4 data/videos_pyramid/

# 1. Quick check on one clip of each kind: are the markers found, and do only the cups and the cap light up?
python scripts/g1_process_videos.py data/videos_pyramid/toss_01.mp4 --check

# 2. Process all clips (the size defaults are your cups and cap; pass yours if they differ). Clips in part_a/,
#    part_b/ and part_c/ folders get their skill from the folder. --focal-px: your phone's focal length in pixels
#    at the clip's resolution (iPhone 16, 1x, 1080x1920: about 1740).
python scripts/g1_process_videos.py data/videos_pyramid/*/*.MOV --focal-px 1740
```

For each clip, open `data/processed/human_pyramid/<clip>_overlay.mp4` and `<clip>_trajectory.png`:

- The yellow path should follow the middle of your palm.
- The palm axes drawn there are: red along the fingers; green pointing away from the thumb; blue out of the
  palm. While you hold a cup, blue should point into it.
- **HOLD** should cover the time you held the object. The white cross marks where the object started; the yellow
  cross marks where the cup ended up, or the top cup's mouth in toss clips.
- Cyan dots with labels mark what was measured in the still frames: `start` / `end` (the centre of the moved
  cup's rim on the table), `base_a` / `base_b` and `mouth` (the pyramid, in top-cup clips), `other` (cup A, when
  cup B is placed). In toss clips the outline of the top cup's opening is drawn too.
- In toss clips, the orange curve is the fitted flight of the cap: it should run through the cap's path.

`summary.json` lists each clip's numbers:

- `scale`: your hand size relative to MediaPipe's average hand. It should be about 0.8–1.25.
- `anchor_err_cm`: how well your hand fits the object at the grasp. A few centimetres is normal.
- `release_dist_cm`, `release_above_mouth_cm`, `off_line_deg`: for throws, where you released the cap relative
  to the top cup's mouth, and how far off the base cups' line (0° = straight along it).
- `release_speed_mps`: for throws, usually 0.5–2 m/s.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Could not find at least 2 ArUco markers" | Make the sheet bigger in the image, improve the lighting, or keep it flat. |
| Something else lights up in `--check`, or a cup has holes | Remove other red/blue things from view; avoid strong reflections on the cups. |
| "Could not tell where the moved cup started / ended" | Your hand was in view in the first or last second, or the cup's start and end spots overlap in the image. |
| "The object never visibly moves" | The moved cup is hidden by your hand for the whole clip, or the camera moved. |
| "The top cup was not found" | The pyramid must be in view and standing in the first second of a toss clip. |
| No `off_line_deg` in the summary | The two base cups weren't seen apart: they stand one behind the other in the image. Turn the pyramid to face the camera. |
| "Could not see the cap leave the hand" / "Too few frames of the cap in flight" | Record at 60 fps from the original file, keep the whole flight in view, and use brighter light. |
| `scale` outside 0.7–1.4 | The focal length estimate is off. Pass `--hfov-deg` (most phone main cameras record video at about 65–75°) or `--focal-px`. |
