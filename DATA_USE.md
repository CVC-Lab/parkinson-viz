# Data use & attribution

This repository visualizes two third-party research datasets. Their use and
redistribution are governed by the data providers' terms, **not** by this repo's code.

## PPMI (Parkinson's Progression Markers Initiative)
- Source: <https://www.ppmi-info.org/>
- The data in `data/merged_data.json` is de-identified and **field-minimized** by
  `convert_data_to_json.py` / `data_pipeline.py`: PATNO plus sex, handedness, enrollment
  age, gait/arm-swing metrics, and clinical scores (UPDRS, Hoehn–Yahr). Birth dates,
  race/ethnicity, visit dates, and enrollment status are **not** exported.
- Use of PPMI data requires agreement to the **PPMI Data Use Agreement** and appropriate
  citation/acknowledgement. **Confirm that public redistribution of these derived fields
  is permitted under your PPMI access before deploying.**

## Synapse WearGait-PD
- Source: <https://www.synapse.org/>
- `data/motion_clips/*.json` are IMU-derived pose clips keyed by Synapse participant IDs
  (no PPMI linkage), carrying age / sex / UPDRS-III / Hoehn–Yahr metadata only.
- Subject to the Synapse WearGait-PD dataset terms and citation requirements.

## Code license
No code `LICENSE` is set yet, so the code is **"all rights reserved"** by default. Add a
LICENSE (e.g. MIT or Apache-2.0) per your institution's policy before others reuse it.
