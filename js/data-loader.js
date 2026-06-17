/**
 * data-loader.js — Loads the pre-processed PPMI dataset and exposes helpers.
 *
 * Adds a derived DUAL_TASK_COST (% gait-speed slowing from single- to dual-task),
 * which is a core biomarker in the source research but was unused before.
 */

export class ParkinsonDataLoader {
    constructor() {
        this.mergedData = [];
    }

    async loadAllDatasets() {
        const response = await fetch('data/merged_data.json');
        if (!response.ok) throw new Error(`Failed to load data: ${response.statusText}`);
        this.mergedData = await response.json();

        // Derived: dual-task gait cost = % slowing in speed under cognitive load.
        for (const row of this.mergedData) {
            row.DUAL_TASK_COST = dualTaskCost(row.SP_U, row.SP__DT);
        }
        return this.mergedData;
    }

    getPatients() {
        return [...new Set(this.mergedData.map(r => r.PATNO))].sort((a, b) => a - b);
    }

    getPatientData(patno) {
        return this.mergedData.find(r => r.PATNO === patno) || null;
    }

    getFeatureStats(feature) {
        const values = this.mergedData
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
        return this.mergedData.filter(r => {
            const x = r[xFeature], y = r[yFeature];
            return x !== null && x !== undefined && !isNaN(x) &&
                   y !== null && y !== undefined && !isNaN(y);
        });
    }

    getCohortCounts() {
        const counts = {};
        this.mergedData.forEach(r => {
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
            'LA_AMP_DT', 'RA_AMP_DT', 'NP3PTRMR', 'NP3PTRML',
        ];
        fields.forEach(f => {
            const s = this.getFeatureStats(f);
            if (s) avg[f] = s.mean;
        });
        return avg;
    }
}

function dualTaskCost(spU, spDT) {
    const a = Number(spU), b = Number(spDT);
    if (!a || isNaN(a) || isNaN(b)) return null;
    return ((a - b) / a) * 100;
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
    OBJECTIVE_MOTOR_SCORE: 'Objective motor impairment',
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
