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

### Real motion from wearable IMU — "WearGait" (2026-06-18)
A new **"WearGait — real motion"** mode plays *measured* motion from the Synapse
WearGait-PD dataset instead of the synthetic gait model.
- `tools/build_motion_clip.py` (offline, conda env) turns a Synapse multi-IMU trial into
  a pose-per-frame clip: arm-swing **shape/phase** from band-passed wrist orientation;
  arm-swing **amount/asymmetry** from frame-invariant **gyro magnitude** (raw L/R Euler
  amplitudes aren't comparable between the two watches — that earlier showed a healthy
  control at 72% "asymmetry"; the gyro metric gives PD ~22% vs Control ~12%); legs timed
  from the real **L/R foot contacts** with knee flexion; seamless stride loop.
- A **sensor-quality gate** flags clips where one wrist barely moved (asymmetry uncertain).
- Each clip is joined to the participant's real **MDS-UPDRS-III / Hoehn-Yahr / age / sex**,
  shown in the summary card alongside the motion.
- 41 clips (24 PD, 17 Control) in `data/motion_clips/` + `index.json`; a picker (shown only
  in WearGait mode) selects among them. PPMI analytics are untouched.
- Limitations: arms/feet/trunk are measured; **knees/hips/elbows are estimated** (real
  timing, synthesized detail) — faithful, not full mocap. The exporter's Synapse data path
  is set via the `SYNAPSE_DIR` env var (with a local default).

### Review pass 2 fixes (2026-06-18)
- **EVENT_ID normalization + no 0-fill severity** (`app.py`): visit IDs are normalized
  (`v8`/`V8` → `V08`) before the UPDRS join, recovering 25 clinical rows; missing severity is
  kept null (not 0). De-dup now runs *before* feature engineering. JSON regenerated: 86
  participants, 79 with real UPDRS-III (was 74); the scatter no longer plots unknowns at 0.
- **Radar Movement axis** (`charts.js`): `MOVEMENT_QUALITY` mapped from its real ~0.6–1.6
  range (was ÷20, which pinned it near zero for everyone).
- **Schematic vs measured labels** (`index.html`/`app.js`): a tag marks the figure
  "Schematic · PPMI metrics" vs "Measured · Synapse WearGait"; subtitle clarifies the two sources.
- **WearGait async guard** (`app.js`): a request token ignores superseded clip fetches.
- **Deploy/repro:** `Procfile` → `gunicorn app:server` (+ `server = app.server`);
  `requirements.txt` → UTF-8; `SYNAPSE_DIR` env var; Roche `SENSOR` composites dropped from the
  Dash axes; stride-time preferred over cadence; README build claim corrected.
- Deferred (lower priority): `convert_data_to_json.py` imports `app.py` (builds Dash at import);
  broad `except` clauses; CDN-only Plotly/Three; clip-JSON schema validation; a11y/mobile polish.

### Pending / next
- More WearGait coverage (HurriedPace / TandemGait, turns); review the 2 quality-flagged clips.
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
