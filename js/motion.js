/**
 * motion.js — Turn a participant's clinical/gait data into a 3D pose.
 *
 * Pure functions (no Three.js): computePose(data, motionType, clock) returns the
 * pose object consumed by Figure3D.applyPose(). All motion is grounded in real
 * PPMI columns so the figure moves like the patient's data, not decoratively:
 *
 *   RA_AMP_U / LA_AMP_U  arm-swing amplitude per side (degrees)  → arm swing + asymmetry
 *   CAD_U                cadence                                  → step frequency
 *   SP_U                 gait speed                               → stride length / leg swing
 *   NP3TOT / NHY         UPDRS-III / Hoehn-Yahr severity          → stoop, reduced amplitude, shuffle
 *   NP3PTRMR / NP3PTRML  postural tremor (R/L)                    → fine hand tremor
 *   SW_PATH_OP / _CL     postural sway path (eyes open/closed)    → balance sway magnitude
 */

const D2R = Math.PI / 180;
const TAU = Math.PI * 2;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (t) => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };

function num(data, key, dflt) {
    if (!data) return dflt;
    const v = data[key];
    if (v === undefined || v === null || Number.isNaN(Number(v))) return dflt;
    return Number(v);
}

// ── Derived gait parameters ────────────────────────────────────────────────
export function strideFrequency(data) {
    const cad = num(data, 'CAD_U', 0);            // cadence (steps/min) — primary, most robust
    if (cad > 0) return clamp(cad / 120, 0.45, 1.2);   // strides per second
    const strideT = num(data, 'STR_T_U', 0);      // fallback: measured stride time (s)
    if (strideT > 0.3) return clamp(1 / strideT, 0.45, 1.2);
    return 0.8;
}

/** Current gait-cycle phase in [0, 2π) for a given clock (seconds). */
export function gaitPhase(data, clock) {
    return (clock * strideFrequency(data) * TAU) % TAU;
}

function severity(data) {
    // Use the raw motor-exam scores only — NOT the 0-filled CLINICAL_MOTOR_SEVERITY —
    // so a participant with no exam stays neutral instead of looking measured-normal.
    const np3 = num(data, 'NP3TOT', 0);
    const hy = num(data, 'NHY', 0);
    return clamp(np3 / 38 + hy / 12, 0, 1.25);      // 0 (none) .. ~1.2 (severe)
}

/** Half-amplitude (radians) of each arm's swing from its measured degrees. */
function armAmps(data) {
    const r = num(data, 'RA_AMP_U', 26);
    const l = num(data, 'LA_AMP_U', 26);
    return {
        r: clamp((r * D2R) / 2, 0.03, 0.55),
        l: clamp((l * D2R) / 2, 0.03, 0.55),
    };
}

/** Which side is more affected (smaller arm swing), and how strongly (0..1). */
export function affectedArm(data) {
    const r = num(data, 'RA_AMP_U', 26);
    const l = num(data, 'LA_AMP_U', 26);
    const denom = r + l + 1e-6;
    const asym = Math.abs(r - l) / denom;          // 0 symmetric .. 1 extreme
    return { side: r < l ? 'r' : 'l', amount: clamp(asym, 0, 1) };
}

// ── Gait / walking ─────────────────────────────────────────────────────────
function gaitPose(data, clock, tc) {
    const th = clock * strideFrequency(data) * TAU;
    const sev = severity(data);
    const amps = armAmps(data);

    const speed = num(data, 'SP_U', 1.05);
    const speedN = clamp(speed / 1.2, 0.25, 1.15);
    const legAmp = (0.18 + 0.20 * speedN) * (1 - 0.30 * sev);   // shuffling when severe
    const kneeAmp = (0.55 + 0.25 * speedN) * (1 - 0.25 * sev);

    // Legs: right leads at θ=0; contralateral arms.
    const hipR = legAmp * Math.sin(th);
    const hipL = legAmp * Math.sin(th + Math.PI);
    const kneeR = 0.06 + kneeAmp * Math.max(0, Math.sin(th - Math.PI / 2));
    const kneeL = 0.06 + kneeAmp * Math.max(0, Math.sin(th + Math.PI / 2));
    const ankleR = -hipR * 0.35 + 0.05;
    const ankleL = -hipL * 0.35 + 0.05;

    // Arms swing opposite their same-side leg, scaled by measured amplitude.
    const shoR = -amps.r * Math.sin(th) * (1 - 0.15 * sev);
    const shoL = amps.l * Math.sin(th) * (1 - 0.15 * sev);
    const elbowSwing = 0.18 + 0.12 * speedN;
    const elbR = elbowSwing * (0.5 + 0.5 * Math.sin(th + Math.PI / 2));
    const elbL = elbowSwing * (0.5 + 0.5 * Math.sin(th - Math.PI / 2));

    // Trunk: stooped with severity, gentle counter-rotation + vertical bob.
    const stoop = 0.05 + 0.20 * sev;
    const chestTwist = -0.09 * Math.sin(th) * (1 - 0.4 * sev);
    const bob = 0.012 * Math.cos(2 * th) - 0.01;
    const headSteady = -stoop * 0.5;               // keep gaze forward despite stoop

    // Postural hand tremor superimposed on the hands (uses wall-clock tc so its frequency
    // does NOT track the playback-speed slider).
    const trem = tremor(data, tc);

    return {
        root: { bob, turn: 0 },
        spine: stoop,
        chestTwist,
        neck: headSteady,
        head: 0,
        arms: {
            l: { shoulder: shoL, elbow: elbL, tremor: trem.l, tremorZ: trem.l * 0.6 },
            r: { shoulder: shoR, elbow: elbR, tremor: trem.r, tremorZ: trem.r * 0.6 },
        },
        legs: {
            l: { hip: hipL, knee: kneeL, ankle: ankleL },
            r: { hip: hipR, knee: kneeR, ankle: ankleR },
        },
    };
}

function tremor(data, tc) {
    const fr = num(data, 'NP3PTRMR', 0);            // postural-tremor scores (UPDRS 3.15)
    const fl = num(data, 'NP3PTRML', 0);
    const osc = Math.sin((tc || 0) * TAU * 5.0);    // fixed ~5 Hz (wall-clock; speed-independent)
    return {
        r: (fr * 1.4 * D2R) * osc,
        l: (fl * 1.4 * D2R) * osc,
    };
}

// ── Postural sway / balance ────────────────────────────────────────────────
function balancePose(data, clock, tc) {
    // Cohort-normalized sway (raw SW_PATH_OP ~1.7–15 is not a metres value, and
    // dividing by 6000 floored every participant to the same minimum).
    const norm = num(data, 'SWAY_NORM', 0.3);
    const mag = 0.012 + 0.05 * clamp(norm, 0, 1);   // ~0.012–0.062 m of CoM sway
    const swayML = mag * Math.sin(clock * TAU * 0.22);
    const swayAP = mag * Math.cos(clock * TAU * 0.17);
    const trem = tremor(data, tc);
    return {
        root: {
            sway: swayML, swayZ: swayAP,
            leanML: swayML * 1.2, leanAP: swayAP * 1.2,
            bob: -0.004,
        },
        spine: 0.05 + 0.5 * Math.abs(swayAP),
        chestTwist: 0,
        // gentle standing stance, slight knee bend, compensatory arm drift
        arms: {
            l: { shoulder: 0.05, elbow: 0.12, shoulderOut: 0.10, tremor: trem.l },
            r: { shoulder: 0.05, elbow: 0.12, shoulderOut: 0.10, tremor: trem.r },
        },
        legs: {
            l: { hip: -0.02, knee: 0.10, ankle: 0.04 - swayAP },
            r: { hip: -0.02, knee: 0.10, ankle: 0.04 - swayAP },
        },
    };
}

// ── Timed Up & Go: sit → stand → walk → turn → walk → sit ───────────────────
const SIT = {
    root: { bob: -0.26 },
    spine: 0.22, neck: -0.12,
    arms: { l: { shoulder: -0.15, elbow: 0.55 }, r: { shoulder: -0.15, elbow: 0.55 } },
    legs: { l: { hip: -1.35, knee: 1.45, ankle: 0.2 }, r: { hip: -1.35, knee: 1.45, ankle: 0.2 } },
};
const STAND = {
    root: { bob: 0 }, spine: 0.06, neck: -0.03,
    arms: { l: { shoulder: 0.02, elbow: 0.14 }, r: { shoulder: 0.02, elbow: 0.14 } },
    legs: { l: { hip: 0, knee: 0.06, ankle: 0.03 }, r: { hip: 0, knee: 0.06, ankle: 0.03 } },
};

function blendPose(a, b, t) {
    t = smooth(t);
    return {
        root: { bob: lerp(a.root?.bob || 0, b.root?.bob || 0, t), turn: lerp(a.root?.turn || 0, b.root?.turn || 0, t) },
        spine: lerp(a.spine || 0, b.spine || 0, t),
        neck: lerp(a.neck || 0, b.neck || 0, t),
        chestTwist: lerp(a.chestTwist || 0, b.chestTwist || 0, t),
        arms: {
            l: blendJoint(a.arms?.l, b.arms?.l, t),
            r: blendJoint(a.arms?.r, b.arms?.r, t),
        },
        legs: {
            l: blendJoint(a.legs?.l, b.legs?.l, t),
            r: blendJoint(a.legs?.r, b.legs?.r, t),
        },
    };
}
function blendJoint(a, b, t) {
    a = a || {}; b = b || {};
    const keys = ['shoulder', 'elbow', 'shoulderOut', 'tremor', 'hip', 'knee', 'ankle'];
    const o = {};
    for (const k of keys) o[k] = lerp(a[k] || 0, b[k] || 0, t);
    return o;
}

function tugPose(data, clock, tc) {
    const dur = clamp(num(data, 'TUG1_DUR', 12), 8, 30);
    const u = (clock / dur) % 1;                    // 0..1 over one full TUG
    let pose, turn = 0;
    if (u < 0.12) {
        pose = SIT;
    } else if (u < 0.24) {
        pose = blendPose(SIT, STAND, (u - 0.12) / 0.12);
    } else if (u < 0.44) {
        pose = gaitPose(data, clock, tc);
    } else if (u < 0.56) {
        pose = gaitPose(data, clock, tc);
        turn = smooth((u - 0.44) / 0.12) * Math.PI;
    } else if (u < 0.76) {
        pose = gaitPose(data, clock, tc);
        turn = Math.PI;
    } else if (u < 0.88) {
        pose = gaitPose(data, clock, tc);
        turn = lerp(Math.PI, TAU, smooth((u - 0.76) / 0.12));
    } else {
        pose = blendPose(STAND, SIT, (u - 0.88) / 0.12);
    }
    pose = JSON.parse(JSON.stringify(pose));
    pose.root = pose.root || {};
    pose.root.turn = (pose.root.turn || 0) + turn;
    return pose;
}

// ── Free / idle: subtle weight shift + breathing + any tremor ───────────────
function idlePose(data, clock, tc) {
    const trem = tremor(data, tc);
    const sway = 0.02 * Math.sin(clock * TAU * 0.18);
    const breathe = 0.01 * Math.sin(clock * TAU * 0.25);
    return {
        root: { sway, bob: breathe },
        spine: 0.05 + 0.18 * severity(data),
        arms: {
            l: { shoulder: 0.03, elbow: 0.13, shoulderOut: 0.08, tremor: trem.l, tremorZ: trem.l * 0.6 },
            r: { shoulder: 0.03, elbow: 0.13, shoulderOut: 0.08, tremor: trem.r, tremorZ: trem.r * 0.6 },
        },
        legs: {
            l: { hip: -0.01, knee: 0.05, ankle: 0.03 },
            r: { hip: -0.01, knee: 0.05, ankle: 0.03 },
        },
    };
}

// ── Public entry point ─────────────────────────────────────────────────────
export function computePose(data, motionType, clock, tc = clock) {
    switch (motionType) {
        case 'balance': return balancePose(data, clock, tc);
        case 'tug': return tugPose(data, clock, tc);
        case 'weargait':                 // clip not yet loaded / failed → neutral idle, never synthetic gait
        case 'free': return idlePose(data, clock, tc);
        case 'gait':
        default: return gaitPose(data, clock, tc);
    }
}
