/**
 * data-loader.js — Loads the pre-processed PPMI dataset and exposes helpers.
 *
 * Two derived quantities are added at load time:
 *   - DUAL_TASK_COST: % gait-speed slowing single- → dual-task (core to the source research).
 *   - SWAY_NORM:      postural-sway path normalized across the cohort (p5–p95), for the
 *                     balance animation (raw SW_PATH_OP is ~1.7–15, not a metres-scale value).
 *
 * The dataset has multiple rows per participant (different visits, and ON/OFF medication
 * states of the same visit). We collapse to ONE canonical record per participant —
 * prefer a row with a motor exam, latest visit, OFF state — so participant selection is
 * deterministic and the cohort scatters don't double-count people. (app.py does the same
 * at the source; this is a defensive guard that also handles a non-deduped JSON.)
 */

export class ParkinsonDataLoader {
    constructor() {
        this.records = [];   // canonical: one row per participant
    }

    async loadAllDatasets() {
        const response = await fetch('data/merged_data.json');
        if (!response.ok) throw new Error(`Failed to load data: ${response.statusText}`);
        const rows = await response.json();

        for (const r of rows) r.DUAL_TASK_COST = dualTaskCost(r.SP_U, r.SP__DT);
        addSwayNorm(rows);
        this.records = dedupeParticipants(rows);
        return this.records;
    }

    getPatients() {
        return this.records.map(r => r.PATNO).sort((a, b) => a - b);
    }

    getPatientData(patno) {
        return this.records.find(r => r.PATNO === patno) || null;
    }

    getFeatureStats(feature) {
        const values = this.records
            .map(r => r[feature])
            .filter(v => v !== null && v !== undefined && !isNaN(v));
        if (!values.length) return null;
        return {
            mean: values.reduce((a, b) => a + b, 0) / values.length,
            min: Math.min(...values),
            max: Math.max(...values),
            count: values.length,
        };
    }

    getValidDataForFeatures(xFeature, yFeature) {
        return this.records.filter(r => {
            const x = r[xFeature], y = r[yFeature];
            return x !== null && x !== undefined && !isNaN(x) &&
                   y !== null && y !== undefined && !isNaN(y);
        });
    }

    getCohortCounts() {
        const counts = {};
        this.records.forEach(r => {
            const c = r.COHORT_NAME || 'Unknown';
            counts[c] = (counts[c] || 0) + 1;
        });
        return counts;
    }

    getAveragePatientData() {
        const avg = { PATNO: null, COHORT_NAME: 'Cohort average' };
        const fields = [
            'LA_AMP_U', 'RA_AMP_U', 'SP_U', 'SP__DT', 'CAD_U', 'ASA_U',
            'L_JERK_U', 'R_JERK_U', 'MOVEMENT_QUALITY', 'BILATERAL_COORDINATION',
            'CLINICAL_MOTOR_SEVERITY', 'NP3TOT', 'NHY', 'DUAL_TASK_COST',
            'LA_AMP_DT', 'RA_AMP_DT', 'NP3PTRMR', 'NP3PTRML', 'SWAY_NORM', 'TOTAL_JERK',
        ];
        fields.forEach(f => {
            const s = this.getFeatureStats(f);
            if (s) avg[f] = s.mean;
        });
        return avg;
    }
}

// ── derived fields ──────────────────────────────────────────────────────────
function dualTaskCost(spU, spDT) {
    const a = Number(spU), b = Number(spDT);
    if (!a || isNaN(a) || isNaN(b)) return null;
    return ((a - b) / a) * 100;
}

function addSwayNorm(rows) {
    const vals = rows
        .map(r => num(r.SW_PATH_OP))
        .filter(v => v !== null)
        .sort((a, b) => a - b);
    const pct = (q) => vals[Math.max(0, Math.min(vals.length - 1, Math.round(q * (vals.length - 1))))];
    const lo = vals.length ? pct(0.05) : 0;
    const hi = vals.length ? Math.max(pct(0.95), lo + 1e-6) : 1;
    for (const r of rows) {
        const v = num(r.SW_PATH_OP) ?? num(r.SW_PATH_CL);
        r.SWAY_NORM = v === null ? null : Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
    }
}

// ── participant de-duplication ──────────────────────────────────────────────
function dedupeParticipants(rows) {
    const byPatno = new Map();
    for (const r of rows) {
        if (!byPatno.has(r.PATNO)) byPatno.set(r.PATNO, []);
        byPatno.get(r.PATNO).push(r);
    }
    const out = [];
    for (const group of byPatno.values()) {
        out.push(group.reduce((best, r) => (recordKey(r) > recordKey(best) ? r : best)));
    }
    return out.sort((a, b) => a.PATNO - b.PATNO);
}

// Comparable key: prefer exam present > latest visit > OFF state (PDSTATE) > higher UPDRS-III.
// Matches app.py's dedupe_participants policy (the JSON is already de-duped; this is a guard).
function recordKey(r) {
    const np3 = num(r.NP3TOT);
    const off = String(r.PDSTATE ?? '').toUpperCase() === 'OFF' ? 1 : 0;
    return (np3 !== null ? 1 : 0) * 1e12
         + eventRank(r.EVENT_ID) * 1e6
         + off * 1e3
         + (np3 !== null ? np3 : -1);
}

function eventRank(ev) {
    const s = String(ev ?? '').toUpperCase();
    if (s === 'SC') return -2;
    if (s === 'BL') return 0;
    const m = s.match(/V0*(\d+)/);
    if (m) return parseInt(m[1], 10);
    const m2 = s.match(/(\d+)/);
    return m2 ? parseInt(m2[1], 10) : 0.5;
}

function num(v) {
    return (v === null || v === undefined || isNaN(v)) ? null : Number(v);
}

export const FEATURE_LABELS = {
    ASA_U: 'Arm-swing asymmetry',
    SP_U: 'Gait speed (m/s)',
    CAD_U: 'Cadence (steps/min)',
    RA_AMP_U: 'Right arm amplitude',
    LA_AMP_U: 'Left arm amplitude',
    ARM_ASYMMETRY: 'Arm amplitude asymmetry index',
    R_JERK_U: 'Right arm jerk (smoothness)',
    L_JERK_U: 'Left arm jerk (smoothness)',
    TOTAL_JERK: 'Total movement jerk',
    MOVEMENT_QUALITY: 'Movement quality index',
    BILATERAL_COORDINATION: 'Bilateral coordination',
    CLINICAL_MOTOR_SEVERITY: 'UPDRS-III motor score',
    PATIENT_REPORTED_SEVERITY: 'UPDRS-II patient score',
    AGE_ADJUSTED_SEVERITY: 'Age-adjusted motor severity',
    OBJECTIVE_MOTOR_SCORE: 'Objective motor index (exploratory)',
    DUAL_TASK_COST: 'Dual-task gait cost (%)',
    SENSOR_MEAN: 'Digital sensor response (mean)',
};

// Muted, accessible categorical palette — restrained, not "rainbow".
export const COHORT_COLORS = {
    "Parkinson's Disease": '#c2533f',
    PD: '#c2533f',
    'Healthy Control': '#3f7d6e',
    Prodromal: '#c2913e',
    SWEDD: '#4a6fa5',
    Unknown: '#9aa3ad',
};
