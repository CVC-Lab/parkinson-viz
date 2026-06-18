# Project status

_Last updated: 2026-06-16_

## What this branch is
`github-pages` is the **live, static, client-side app** served at
<https://cvc-lab.github.io/parkinson-viz/> (linked from the lab site
<https://cvc-lab.github.io/parkinsons-website/>). It runs entirely in the browser —
no backend. The `master` branch holds the original Python/Dash app (`app.py`).

## Current state (this commit)
Full revamp of the dashboard. The previous "silhouette" — ~16 disconnected, rainbow,
independently-translated polygons that visibly fell apart during animation — has been
**replaced with a real, interactive 3D model**, and the page has been restyled.

### Done
- **3D motion model** (`js/figure3d.js`, Three.js): a single-material articulated
  mannequin built as a true forward-kinematics joint hierarchy, so limbs stay
  connected. Orbit / zoom, Front / Side / ¾ / Top presets + reset, studio lighting,
  contact shadow.
- **Data-driven motion** (`js/motion.js`): gait (per-side arm-swing amplitude →
  visible asymmetry, cadence, gait speed, knee flexion, severity stoop/shuffle,
  hand tremor), postural sway / balance, TUG (sit→stand→walk→turn→sit), free/idle.
  The more-affected arm is tinted amber on the model.
- **Clinical summary card** + metric chips: cohort, age, sex, handedness,
  Hoehn–Yahr, UPDRS-III/II, gait speed, cadence, arm-swing asymmetry,
  **dual-task gait cost** (newly surfaced), tremor.
- **Professional restyle** (`css/styles.css`): restrained palette, real type scale,
  hairline borders, no emoji/rainbow.
- All prior controls and charts retained (`js/app.js`, `js/charts.js`,
  `js/data-loader.js`): patient select, motion type, speed, play/pause/reset,
  X/Y feature selectors, correlation scatter, bilateral asymmetry, gait-cycle,
  quality radar.
- Verified with headless-Chrome / puppeteer screenshots across patients, motion
  types, and camera views. Fixed two real bugs found that way: canvas rendered at
  2× and clipped off-screen; shoulder-abduction sign inverted (arms collapsed inward).

### Data-correctness fixes (2026-06-18 review pass)
- **[1] One record per participant.** `app.py` now de-dups the visit / ON-OFF (`PDSTATE`)
  fan-out → one row per PATNO (exam present, latest visit, OFF state); JSON regenerated
  (190 → 86 rows). `data-loader.js` also de-dups defensively. Stops arbitrary severity and
  cohort-scatter double-counting.
- **[2] Balance is patient-specific.** Sway scales with cohort-normalized `SWAY_NORM`
  (was `path/6000`, which floored every participant to one value).
- **[3] Missing UPDRS = "Unknown", not 0.** `motion.js` drops the 0-fill; the summary
  shows "Unknown" and suppresses the tremor chip when there's no exam (12 participants).
- **[4] Radar symmetry axis** uses `BILATERAL_COORDINATION` (was `1 - ASA_U/2`, which
  collapsed to 0 for 186/190).
- **[5] Docs refreshed** (`README.md`, `README_GITHUB_PAGES.md`).

### Pending / next
- **Two charts to upgrade for substance** (separate from the [4] bugfix):
  gait-cycle waveform → single- vs dual-task comparison; quality radar → real
  UPDRS-III motor-domain profile.
- **Sync `app.py`** fully to the 3D motion model (only the data pipeline is shared today).
- Optional figure polish: taper the torso, refine hands/feet, front-facing default cue.

## Run locally
```bash
python3 -m http.server 8000 --bind 0.0.0.0   # from repo root, then open localhost:8000
```
The 3D model and charts load Three.js / Plotly from CDN (internet needed on first load).
