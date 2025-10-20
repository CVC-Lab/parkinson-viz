/**
 * Parkinson Data Loader - JavaScript Port
 * Loads and processes multi-modal Parkinson's datasets
 */

export class ParkinsonDataLoader {
    constructor() {
        this.data = {};
        this.mergedData = [];
    }

    async loadAllDatasets() {
        console.log('Loading datasets...');

        try {
            // Load merged dataset directly (pre-processed)
            const response = await fetch('data/merged_data.json');
            if (!response.ok) {
                throw new Error(`Failed to load data: ${response.statusText}`);
            }

            this.mergedData = await response.json();
            console.log(`✓ Loaded ${this.mergedData.length} patient records`);

            return this.mergedData;
        } catch (error) {
            console.error('Error loading datasets:', error);
            throw error;
        }
    }

    getPatients() {
        const patients = [...new Set(this.mergedData.map(row => row.PATNO))];
        return patients.sort((a, b) => a - b);
    }

    getPatientData(patno) {
        return this.mergedData.find(row => row.PATNO === patno) || null;
    }

    getFeatureStats(feature) {
        const values = this.mergedData
            .map(row => row[feature])
            .filter(v => v !== null && v !== undefined && !isNaN(v));

        if (values.length === 0) return null;

        return {
            mean: values.reduce((a, b) => a + b, 0) / values.length,
            min: Math.min(...values),
            max: Math.max(...values),
            count: values.length
        };
    }

    getDataByCohort(cohortName) {
        return this.mergedData.filter(row => row.COHORT_NAME === cohortName);
    }

    getValidDataForFeatures(xFeature, yFeature) {
        return this.mergedData.filter(row => {
            const x = row[xFeature];
            const y = row[yFeature];
            return x !== null && x !== undefined && !isNaN(x) &&
                   y !== null && y !== undefined && !isNaN(y);
        });
    }

    getCohortCounts() {
        const cohorts = {};
        this.mergedData.forEach(row => {
            const cohort = row.COHORT_NAME || 'Unknown';
            cohorts[cohort] = (cohorts[cohort] || 0) + 1;
        });
        return cohorts;
    }

    getAveragePatientData() {
        // Calculate average values for all numeric features
        const avg = {};
        const numericFields = [
            'LA_AMP_U', 'RA_AMP_U', 'SP_U', 'ASA_U',
            'L_JERK_U', 'R_JERK_U', 'MOVEMENT_QUALITY',
            'BILATERAL_COORDINATION', 'CLINICAL_MOTOR_SEVERITY'
        ];

        numericFields.forEach(field => {
            const stats = this.getFeatureStats(field);
            if (stats) {
                avg[field] = stats.mean;
            }
        });

        return avg;
    }
}

export const FEATURE_LABELS = {
    'ASA_U': 'Arm Swing Asymmetry',
    'SP_U': 'Gait Speed (m/s)',
    'RA_AMP_U': 'Right Arm Amplitude',
    'LA_AMP_U': 'Left Arm Amplitude',
    'ARM_ASYMMETRY': 'Arm Amplitude Asymmetry Index',
    'R_JERK_U': 'Right Arm Jerk (Smoothness)',
    'L_JERK_U': 'Left Arm Jerk (Smoothness)',
    'JERK_ASYMMETRY': 'Bilateral Jerk Asymmetry',
    'TOTAL_JERK': 'Total Movement Jerk',
    'MOVEMENT_QUALITY': 'Movement Quality Index',
    'BILATERAL_COORDINATION': 'Bilateral Coordination Score',
    'CLINICAL_MOTOR_SEVERITY': 'UPDRS-III Motor Score',
    'PATIENT_REPORTED_SEVERITY': 'UPDRS-II Patient Score',
    'AGE_ADJUSTED_SEVERITY': 'Age-Adjusted Motor Severity',
    'OBJECTIVE_MOTOR_SCORE': 'Objective Motor Impairment',
    'SENSOR_CLINICAL_RATIO': 'Sensor-Clinical Correlation',
    'SENSOR_MEAN': 'Digital Sensor Response (Mean)',
    'SENSOR_STD': 'Digital Sensor Variability',
    'SENSOR_COUNT': 'Digital Assessment Frequency'
};

export const COHORT_COLORS = {
    "Parkinson's Disease": '#e74c3c',
    'PD': '#e74c3c',
    'Healthy Control': '#27ae60',
    'Prodromal': '#f39c12',
    'SWEDD': '#3498db',
    'Unknown': '#95a5a6'
};
