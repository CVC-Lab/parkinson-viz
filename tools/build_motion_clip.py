#!/usr/bin/env python3
"""Build real-motion clips (figure pose per frame) from Synapse WearGait trials.

Synapse multi-IMU CSVs give per-segment orientation (Roll/Pitch/Yaw), triaxial
gyroscope, and binary L/R foot contacts at 100 Hz. Sensor-frame Euler angles are NOT
anatomical joint angles (drift; Yaw wraps; and the two wrist watches are mounted
differently so their Euler amplitudes are not comparable L-vs-R). So:

  - Arm-swing SHAPE / phase: band-pass each wrist's orientation around the gait
    frequency (normalized to unit amplitude — shape only).
  - Arm-swing AMOUNT / asymmetry: RMS of the gait-band GYRO MAGNITUDE, which is
    rotation-rate and therefore frame-invariant (comparable between wrists). The
    figure's two arms are scaled by this amount, so the rendered L/R asymmetry is
    real, and asymmetry% is computed from it.
  - Quality gate: if a wrist's gait-band gyro amount is below a floor (weak / dropped
    sensor), the clip's asymmetry is flagged unreliable.
  - Legs: timed from the real L/R foot contacts (heel-strike -> heel-strike); hip
    swing + double-bump knee flexion follow that real timing.
  - Clinical: each participant is joined to their MDS-UPDRS-III / Hoehn-Yahr / age / sex.

Output: one pose-per-frame JSON per participant + an index.json manifest.
"""
import json, os, glob, re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend

SYN = os.environ.get(
    "SYNAPSE_DIR",
    "/Users/ryanfarell/repos-local/parkinsons-project/Gait Data/Synapse_Wear-Gait_PD",
)
OUT_DIR = "data/motion_clips"
TARGET_FPS = 30
D2R = np.pi / 180
GYRO_FLOOR = 0.2          # rad/s gait-band gyro RMS; below this one wrist barely moved,
                          # so the asymmetry % rests on a small denominator (flag as uncertain)


def load_walk(path):
    df = pd.read_csv(path, low_memory=False)
    walk = df[df["GeneralEvent"].astype(str) == "Walk"].reset_index(drop=True)
    t = walk["Time"].astype(str).str.replace(" sec", "", regex=False).astype(float).values
    fs = 1.0 / np.median(np.diff(t))
    return walk, t, fs


def col(walk, name):
    if name not in walk:
        return None
    return pd.to_numeric(walk[name], errors="coerce").interpolate(limit_direction="both").fillna(0).values


def bandpass(x, fs, f0):
    lo, hi = max(0.4, f0 * 0.5), min(fs / 2 - 0.1, f0 * 2.5)
    b, a = butter(2, [lo, hi], btype="band", fs=fs)
    return filtfilt(b, a, detrend(x))


def best_swing(walk, fs, f0, seg):
    best, bestp = None, -1.0
    for ax in ("Roll", "Pitch"):
        x = col(walk, f"{seg}_{ax}")
        if x is None:
            continue
        y = bandpass(x, fs, f0)
        if np.var(y) > bestp:
            bestp, best = np.var(y), y
    return best


def gyro_amount(walk, fs, f0, seg, win):
    """Frame-invariant arm-swing amount: RMS of gait-band gyro magnitude (rad/s)."""
    g = [col(walk, f"{seg}_Gyr_{a}") for a in "XYZ"]
    if any(x is None for x in g):
        return 0.0
    mag = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)[win]
    return float(np.sqrt(np.mean(bandpass(mag, fs, f0) ** 2)))


def heel_strikes(contact):
    b = (np.nan_to_num(contact) > 0.5).astype(int)
    return np.where(np.diff(b) == 1)[0] + 1


def phase_series(hs, n):
    phi = np.zeros(n)
    for k in range(len(hs) - 1):
        a, c = hs[k], hs[k + 1]
        if c > a:
            phi[a:c] = np.linspace(0, 1, c - a, endpoint=False)
    return phi


def clean_window(hs, fs, min_strides=3, max_dur=8.0):
    if len(hs) < min_strides + 1:
        return hs[0], hs[-1]
    d = np.diff(hs) / fs
    med = np.median(d)
    best, i = (0, 1), 0
    while i < len(d):
        j = i
        while j < len(d) and d[j] < 1.6 * med:
            j += 1
        if hs[j] - hs[i] > hs[best[1]] - hs[best[0]]:
            best = (i, j)
        i = max(j + 1, i + 1)
    a, b = best
    while (hs[b] - hs[a]) / fs > max_dur and b > a + min_strides:
        b -= 1
    return hs[a], hs[b]


def knee_curve(phi):
    loading = 0.18 * np.exp(-((phi - 0.12) / 0.08) ** 2)
    swing = 0.95 * np.exp(-((phi - 0.80) / 0.10) ** 2)
    return 0.06 + loading + swing


def load_clinical():
    """Per-subject MDS-UPDRS-III / Hoehn-Yahr / age / sex, keyed by subject id."""
    clin = {}
    for fn in ("PD - Demographic+Clinical - datasetV1.csv",
               "CONTROLS - Demographic+Clinical - datasetV1.csv"):
        p = os.path.join(SYN, fn)
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, header=1, low_memory=False)
        except Exception:
            continue
        cols = {str(c).strip(): c for c in df.columns}
        idc = cols.get("Subject ID")
        if not idc:
            continue
        u3 = [c for c in df.columns if str(c).startswith("MDSUPDRS_3-")]
        hy, age, sex = cols.get("Modified Hoehn & Yahr Score"), cols.get("Age (years)"), cols.get("Sex")
        sid = df[idc].astype(str).str.replace(r"\s*\(.*$", "", regex=True).str.strip()
        for i in range(len(df)):
            s = sid.iloc[i]
            if not s or s.lower() == "nan":
                continue
            t = pd.to_numeric(df.iloc[i][u3], errors="coerce") if u3 else pd.Series([], dtype=float)
            def g(c):
                if not c:
                    return None
                v = pd.to_numeric(df.iloc[i][c], errors="coerce")
                return None if pd.isna(v) else v
            hyv, agev = g(hy), g(age)
            sexv = df.iloc[i][sex] if sex else None
            clin[s] = {
                "updrs3": int(t.sum()) if t.notna().sum() >= 20 else None,
                "hy": None if hyv is None else float(hyv),
                "age": None if agev is None else int(agev),
                "sex": None if (sexv is None or str(sexv) == "nan") else str(sexv),
            }
    return clin


def build(path, cohort, clin):
    pid = os.path.basename(path).split("_")[0].split(" (")[0].strip()
    walk, t, fs = load_walk(path)
    n = len(walk)

    hsL = heel_strikes(col(walk, "L Foot Contact"))
    hsR = heel_strikes(col(walk, "R Foot Contact"))
    if len(hsL) < 4:
        raise ValueError("too few left heel strikes")   # ValueError so batch() skips, not aborts
    # Stride frequency from the real heel-strike interval. (An FFT global-max over 0.5–2 Hz
    # locked onto the step harmonic for a few clips → impossible cadence; clamp to physiologic.)
    f0 = float(np.clip(fs / np.median(np.diff(hsL)), 0.4, 1.4))
    s, e = clean_window(hsL, fs)
    win = slice(s, e)
    phiL = phase_series(hsL, n)[win]
    # Right-leg timing must cover the (left-derived) window; else fall back to contralateral.
    hsR_in = hsR[(hsR >= s) & (hsR < e)]
    if len(hsR_in) >= 2:
        phiR, leg_quality = phase_series(hsR, n)[win], "ok"
    else:
        phiR, leg_quality = (phiL + 0.5) % 1.0, "synth-right"

    LEG_AMP = 0.42
    hipL, hipR = -LEG_AMP * np.cos(2 * np.pi * phiL), -LEG_AMP * np.cos(2 * np.pi * phiR)
    kneeL, kneeR = knee_curve(phiL), knee_curve(phiR)
    ankleL, ankleR = 0.12 * np.cos(2 * np.pi * phiL), 0.12 * np.cos(2 * np.pi * phiR)

    # Arm shape (orientation, normalized) + frame-invariant amount (gyro)
    waveL = best_swing(walk, fs, f0, "L_Wrist")[win]
    waveR = best_swing(walk, fs, f0, "R_Wrist")[win]
    waveL /= np.percentile(np.abs(waveL), 95) + 1e-9
    waveR /= np.percentile(np.abs(waveR), 95) + 1e-9
    if np.corrcoef(waveL, hipL)[0, 1] > 0:
        waveL = -waveL
    if np.corrcoef(waveR, hipR)[0, 1] > 0:
        waveR = -waveR
    amtL = gyro_amount(walk, fs, f0, "L_Wrist", win)
    amtR = gyro_amount(walk, fs, f0, "R_Wrist", win)
    asym = abs(amtL - amtR) / (amtL + amtR + 1e-9)
    quality = "ok" if min(amtL, amtR) >= GYRO_FLOOR else "low"
    big = max(amtL, amtR) + 1e-9
    shoL = waveL * (24 * D2R) * (amtL / big)
    shoR = waveR * (24 * D2R) * (amtR / big)

    trunk = bandpass(col(walk, "LowerBack_Roll"), fs, f0)[win]
    chest = bandpass(col(walk, "Xiphoid_Roll"), fs, f0)[win]
    head = bandpass(col(walk, "Forehead_Roll"), fs, f0)[win]
    trunk *= (4 * D2R) / (np.percentile(np.abs(trunk), 95) + 1e-9)
    chest *= (4 * D2R) / (np.percentile(np.abs(chest), 95) + 1e-9)
    head *= (4 * D2R) / (np.percentile(np.abs(head), 95) + 1e-9)
    bob = -0.012 * np.cos(2 * np.pi * 2 * phiL) - 0.008

    dur = (e - s) / fs
    m = max(2, int(round(dur * TARGET_FPS)))
    idx = np.linspace(0, (e - s) - 1, m).astype(int)
    fr = lambda a, i: float(a[idx[i]])

    frames = []
    for i in range(m):
        frames.append({
            "root": {"leanML": fr(trunk, i), "bob": fr(bob, i)},
            "spine": 0.06, "chestTwist": fr(chest, i), "head": fr(head, i),
            "arms": {
                "l": {"shoulder": fr(shoL, i), "elbow": 0.14 + 0.22 * abs(fr(shoL, i))},
                "r": {"shoulder": fr(shoR, i), "elbow": 0.14 + 0.22 * abs(fr(shoR, i))},
            },
            "legs": {
                "l": {"hip": fr(hipL, i), "knee": fr(kneeL, i), "ankle": fr(ankleL, i)},
                "r": {"hip": fr(hipR, i), "knee": fr(kneeR, i), "ankle": fr(ankleR, i)},
            },
        })

    c = clin.get(pid, {})
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {
        "id": pid, "cohort": cohort, "source": "Synapse WearGait-PD (SelfPace)",
        "fps": TARGET_FPS, "durationS": round(dur, 1), "gaitHz": round(f0, 2),
        "asymmetryPct": round(asym * 100), "armQuality": quality, "legQuality": leg_quality,
        "armAmtL": round(amtL, 2), "armAmtR": round(amtR, 2),
        "updrs3": c.get("updrs3"), "hy": c.get("hy"), "age": c.get("age"), "sex": c.get("sex"),
    }
    json.dump({**meta, "frames": frames}, open(os.path.join(OUT_DIR, f"{pid}.json"), "w"))
    print(f"  {pid} ({cohort}): asym {meta['asymmetryPct']}% [{quality}] "
          f"amt {amtL:.1f}/{amtR:.1f} UPDRS3={c.get('updrs3')} HY={c.get('hy')}")
    return meta


def batch():
    clin = load_clinical()
    print(f"clinical records: {len(clin)}")
    manifest = []
    for cohort, sub in (("PD", "PD_PARTICIPANTS"), ("Control", "CONTROL_PARTICIPANTS")):
        files = sorted(glob.glob(os.path.join(SYN, sub, "CSV files", "*SelfPace*.csv")))
        files = [f for f in files if "TURN" not in os.path.basename(f)]
        seen = set()
        for f in files:
            pid = os.path.basename(f).split("_")[0].split(" (")[0].strip()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                manifest.append(build(f, cohort, clin))
            except Exception as ex:
                print(f"  skip {pid}: {ex}")
    # good quality first, then by cohort, then by asymmetry
    manifest.sort(key=lambda m: (m["armQuality"] != "ok", m["cohort"] != "PD", -m["asymmetryPct"]))
    json.dump({"clips": manifest}, open(os.path.join(OUT_DIR, "index.json"), "w"), indent=1)
    ok = sum(1 for m in manifest if m["armQuality"] == "ok")
    print(f"\nmanifest: {len(manifest)} clips ({ok} good arm-signal) -> {OUT_DIR}/index.json")


if __name__ == "__main__":
    batch()
