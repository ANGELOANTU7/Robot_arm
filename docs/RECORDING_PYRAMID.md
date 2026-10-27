# Recording the cup-pyramid skills

The G1 learns the pyramid task from short phone clips of **one skill each**:

| Clip | What you do | How many |
|---|---|---|
| `place_base_NN.mp4` | Take a mouth-down cup from the supply area and set it down mouth-down at its pyramid spot. | ~10 (cup A on its own, then cup B next to A) |
| `place_top_NN.mp4` | Take the mouth-up top cup and set it centred on the two base cups. | ~10 |
| `toss_NN.mp4` | Pick the ball up from the table, turn your palm up and toss it underhand into the top cup. | ~10 |

One fixed camera is enough. The printed marker sheet gives each clip a metric table frame. The pipeline
recovers your **palm's 3D position and orientation**, how closed your hand is, the moment you grasp and release,
where the cup started and ended, and, for tosses, the ball's flight.

## What you need

| Item | Notes |
|---|---|
| Phone + tripod (or a stack of books) | The camera must not move during a clip. |
| `docs/marker_sheet.pdf` printed on A4 | Print at **100% scale**. Measure a black square: it should be 50 mm (else pass `--marker-mm`). |
| 3 plastic cups of the **same size** in **3 different solid colors** | Defaults: cup A **blue**, cup B **green**, top cup **yellow** (`--cup-a-color`, `--cup-b-color`, `--top-color`). Measure the height and the rim's outer diameter. |
| A 4 cm **orange** ping-pong ball | `--ball-color orange`; `--ball-diameter-cm` if yours differs. |
| A plain table | It must not be any of those colors. Keep other objects of those colors out of view. |

Use your **right hand** only. The robot uses its right hand, and the palm orientation assumes a right hand.

## Setup (seen from above)

```
                        you (reach in from this side)
        ┌────────────────────────────────────────────────────┐
        │  supply (and the ball         pyramid spot         │
        │  in toss clips):                                   │
        │  ▲A  ▲B  ▽top  ●ball              ▲▲               │  ← your right is on the left of this picture
        │                                                    │
        │              ┌───────────────────┐                 │
        │              │  ▣             ▣  │                 │  ← marker sheet, flat, between the camera
        │              │        y↑ →x      │                 │    and the work area
        │              │  ▣             ▣  │                 │
        │              └───────────────────┘                 │
        └────────────────────────────────────────────────────┘
                                📱 camera
                  80–100 cm away, 30–45° above the table
```

(▲ = mouth down, ▽ = mouth up.)

- The camera must see the whole sheet, all cups, the ball and **your whole hand path**. That includes the
  hand lifted above the cups and, in toss clips, the wind-up and the ball's whole flight. Sit it further back
  and a little higher than for the pinch clips.
- Use landscape, 1080p and the normal 1× lens. Record at **60 fps**; it is needed for the tosses, and fine for the
  others.
- Use bright, even light. Short exposure keeps the flying ball sharp, and the ball's size in the image gives
  its depth.
- Lock focus and exposure if your phone allows it.

## Each clip (about 4–8 seconds)

Always start and end by **holding still for 1 second** with your hand out of the frame or well away. The first
second calibrates the camera from the sheet.

### place_base

1. Grab the cup **from the side with your whole hand**, like holding a glass: thumb up, palm against the cup,
   about halfway up. Don't grab it from above.
2. Lift it about 10 cm, carry it over its spot and lower it until it stands on the table, mouth down.
3. Open your hand and pull it back.

Record cup A alone at the pyramid spot. Then record cup B being placed **next to A, almost touching it** (a
gap of a few millimetres). Do both several times, with the supply cups and the pyramid spot moved a little
between clips.

### place_top

1. Start with A and B standing side by side, mouth down. The top cup stands mouth up in the supply area.
2. Grab the top cup from the side, lift it over the pair and set it down **centred across both base cups**.
3. Open your hand gently and pull it back. The pyramid must stay standing.

### toss

1. Start with the pyramid built and the ball lying on the table to your right (pass `--ball-rest-height-cm` if
   it sits on a bottle cap or similar).
2. Pick the ball up, turn your **palm up** with the ball resting in it, and wind up beside the pyramid.
3. Toss it **underhand** from about 20–30 cm away, from your right-hand side, so the ball flies across the
   camera's view and not towards or away from it. Let it drop into the top cup.

Misses and collapsing pyramids are fine to record, but successful throws teach the most. The robot copies your
throw's style (wind-up, release height and arc), and its aim is corrected in simulation.

## Processing

```bash
mkdir -p data/videos_pyramid && cp /path/to/clips/*.mp4 data/videos_pyramid/

# 1. Quick check on one clip of each kind: are the markers found, and is each object one solid patch?
python scripts/g1_process_videos.py data/videos_pyramid/toss_01.mp4 --check

# 2. Process all clips (use your measured sizes)
python scripts/g1_process_videos.py data/videos_pyramid/*.mp4 --cup-height-cm 7.5 --cup-diameter-cm 6.2
```

For each clip, open `data/processed/human_pyramid/<clip>_overlay.mp4` and `<clip>_trajectory.png`:

- The yellow path should follow the middle of your palm.
- The palm axes drawn there are: red along the fingers; green pointing away from the thumb; blue out of the
  palm. While you hold a cup, blue should point into it.
- **HOLD** should cover the time you held the object. The white cross marks where the object started; the
  yellow cross marks where the cup ended up, or the top cup's mouth in toss clips.

`summary.json` lists each clip's numbers:

- `scale`: your hand size relative to MediaPipe's average hand. It should be about 0.8–1.25.
- `anchor_err_cm`: how well the palm fits the object at the grasp. A few centimetres is normal.
- `release_speed_mps`: for tosses, usually 1–2.5 m/s.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Could not find at least 2 ArUco markers" | Make the sheet bigger in the image, improve the lighting, or keep it flat. |
| An object isn't detected, or the wrong pixels light up in `--check` | Use a color that stands out from the table. Pick another preset (`blue`, `green`, `yellow`, `pink`, `red`, `orange`). |
| "The object never visibly moves" | The wrong cup color is set, or the moved cup is hidden by your hand for the whole clip. |
| "Could not see the ball leave the hand" / "Too few frames of the ball in flight" | Record at 60 fps, keep the whole flight in view, and use brighter light. |
| `scale` outside 0.7–1.4 | The focal length estimate is off. Pass `--hfov-deg` (most phone main cameras record video at about 65–75°) or `--focal-px`. |
