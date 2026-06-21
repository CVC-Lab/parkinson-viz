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
  participants, 78 with real UPDRS-III (was 74); the scatter no longer plots unknowns at 0.
- **Radar Movement axis** (`charts.js`): `MOVEMENT_QUALITY` mapped from its real ~0.6–1.6
  range (was ÷20, which pinned it near zero for everyone).
- **Schematic vs measured labels** (`index.html`/`app.js`): a tag marks the figure
  "Schematic · PPMI metrics" vs "Measured · Synapse WearGait"; subtitle clarifies the two sources.
- **WearGait async guard** (`app.js`): a request token ignores superseded clip fetches.
- **Deploy/repro:** `Procfile` → `gunicorn app:server` (+ `server = app.server`);
  `requirements.txt` → UTF-8; `SYNAPSE_DIR` env var; Roche `SENSOR` composites dropped from the
  Dash axes; stride-time preferred over cadence; README build claim corrected.
- **Pipeline reproducibility:** the data loader is extracted to `data_pipeline.py`, so
  `convert_data_to_json.py` reuses it **without building the Dash app** (JSON byte-identical
  after the refactor).
- Still deferred (lower priority): broad `except` clauses in the loaders; CDN-only Plotly/Three;
  clip-JSON schema validation; a11y/mobile polish.

### Debate-pass fixes (2026-06-19, multi-agent review)
- **#9 (was the worst live bug):** `drawAllCharts()` is now mode-aware — changing the PPMI
  participant in WearGait mode no longer repaints the synthetic sine over the measured waveform.
- **#7 honesty:** WearGait labels softened to "IMU-derived" (timing + arm-swing measured;
  knees/trunk/joints estimated) — subtitle, mode tag, option, source row, figure caption, + a
  "Provenance" summary row.
- **#3 disclosure:** participant card shows **Medication state (ON/OFF/Unknown)** next to UPDRS-III.
- **#6:** `strideFrequency` uses cadence as primary (stride-time only as fallback).
- **#8:** "Objective motor impairment" → "Objective motor index (exploratory)".
- **#2:** date-tolerance guard in the pipeline nulls UPDRS joins with a >6-month gait/exam gap
  (fixes PATNO 40611's 4.3-yr-stale exam).
- **#1 / #4 privacy + export hygiene:** `convert_data_to_json.py` drops birth dates, race flags,
  visit dates, enrollment status, and the incoherent Roche `SENSOR_*` composites (110→90 fields);
  README privacy/cohort/clinical claims corrected; added `CITATION.cff` + `DATA_USE.md`.
- Deferred: per-field availability flags (#5, 1–3 participants), a code `LICENSE` (org decision),
  `app.py` Dash-UI parity, the unused `SEVERITY_CATEGORY` legacy column.

### Full-app review fixes (2026-06-21, 53-agent review; 0 high/critical found)
- **WearGait integrity:** figure no longer shows synthetic gait under the IMU tag while a clip
  loads or if a fetch fails (`computePose` has a `weargait`→idle case; mode tag now cycles
  Loading→IMU-derived / "IMU clip unavailable"). Entering WearGait or changing the PPMI
  participant no longer leaves that participant highlighted on the cohort charts (`selected=null`).
- **Tremor fidelity:** tremor runs at a fixed ~5 Hz off an unscaled wall-clock (was 5 Hz ×
  speed-slider). Chip relabeled "Postural tremor (UPDRS 3.15)" — only postural items exist.
- **Radar:** dropped the "Movement" axis (~collinear with Speed, r≈0.92); rescaled Smoothness
  so it no longer pins ~14% of participants at 0. Removed the weak `OBJECTIVE_MOTOR_SCORE`
  (r=0.34) from the scatter axis options.
- **Build pipeline:** stride frequency from the real heel-strike interval (clamped 0.4–1.4 Hz)
  — kills the FFT harmonic lock-on that showed 209–233 steps/min for 3 control clips (now
  106–137); `SystemExit`→`ValueError` so one bad trial skips instead of aborting the batch
  (41→42 clips); right-leg timing validated with a contralateral fallback + `legQuality`.
  Regenerated all 42 clips.
- **A11y / docs:** `prefers-reduced-motion` starts paused + disables CSS transitions; READMEs
  de-overclaim "Digital Sensor" data, fix clone/Live-Demo URLs; STATUS UPDRS-III count 79→78;
  `FIXES_APPLIED.md` marked historical.
- Deferred (low, remaining): asym-chip units, negative-dual-task rescale, on-demand rendering,
  build-time Control UPDRS-III plausibility flag, UPDRS-II staleness guard, SEX/HANDED legend.

### Polish pass (2026-06-21)
- **Elbow direction fixed** (user-reported): the forearm now flexes forward (+Z) — `elbow.rotation.x`
  applied +X, which hyperextended it backward. Also raised the figure so the feet rest on the ground.
- WebGL context-loss/restore handler; `realWaveform` & `loadManifest` guard short/missing clips;
  radar + gait-cycle charts label the cohort-average case and tag the waveform "schematic"; speed
  slider `aria-valuetext`, status pill `role=status`/`aria-live`, Playback/Status groups.
- **Unified source selector** (user feedback): one contextual dropdown in the top-left slot —
  "Participant" (PPMI) in the schematic modes, "WearGait recording" (the 42 Synapse clips) in
  WearGait — instead of a greyed-out Participant box plus a separate bottom clip picker.
- **"How to read this" info overlay** (user feedback): a header button opens a modal (dimmed/blurred
  backdrop, centered dialog; closes via ✕ / backdrop / Esc) explaining schematic-vs-IMU motion, the
  five motion tests, the 3D controls, the four charts, and what to look for.

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
