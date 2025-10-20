/**
 * Multi-Modal Parkinson's Disease Motion Visualization
 * Main Application Entry Point
 */

import { ParkinsonDataLoader, FEATURE_LABELS, COHORT_COLORS } from './data-loader.js';
import { MotionSilhouetteGenerator } from './motion-generator.js';

// Global state
const state = {
    dataLoader: null,
    motionGenerator: null,
    selectedPatient: null,
    currentPatientData: null,
    animationState: {
        playing: true,
        timePhase: 0,
        speed: 1.0
    },
    animationFrameId: null
};

// Plotly configuration for all charts
const plotlyConfig = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
};

// Initialize application
async function init() {
    console.log('🚀 Initializing Parkinson\'s Motion Visualization...');

    // Initialize data loader and motion generator
    state.dataLoader = new ParkinsonDataLoader();
    state.motionGenerator = new MotionSilhouetteGenerator();

    try {
        // Load data
        await state.dataLoader.loadAllDatasets();

        // Initialize with average patient data
        state.currentPatientData = state.dataLoader.getAveragePatientData();

        // Populate patient dropdown
        populatePatientDropdown();

        // Set up event listeners
        setupEventListeners();

        // Initialize visualizations with default data
        updateAllVisualizations();

        // Start animation loop
        startAnimation();

        console.log('✅ Application initialized successfully!');
    } catch (error) {
        console.error('❌ Error initializing application:', error);
        alert('Error loading data. Please check the console for details.');
    }
}

function populatePatientDropdown() {
    const patients = state.dataLoader.getPatients();
    const select = document.getElementById('patient-select');

    select.innerHTML = '<option value="">Select a patient...</option>';

    patients.forEach(patno => {
        const option = document.createElement('option');
        option.value = patno;
        option.textContent = `Patient ${patno}`;
        select.appendChild(option);
    });
}

function setupEventListeners() {
    // Patient selection
    document.getElementById('patient-select').addEventListener('change', (e) => {
        const patno = parseInt(e.target.value);
        if (patno) {
            state.selectedPatient = patno;
            state.currentPatientData = state.dataLoader.getPatientData(patno);
        } else {
            state.selectedPatient = null;
            state.currentPatientData = state.dataLoader.getAveragePatientData();
        }
        updateAllVisualizations();
    });

    // Motion test type
    document.getElementById('motion-test-select').addEventListener('change', () => {
        updateAllVisualizations();
    });

    // Animation speed
    const speedSlider = document.getElementById('animation-speed');
    const speedDisplay = document.getElementById('speed-display');
    speedSlider.addEventListener('input', (e) => {
        state.animationState.speed = parseFloat(e.target.value);
        speedDisplay.textContent = `${state.animationState.speed.toFixed(1)}x`;
    });

    // Animation controls
    document.getElementById('play-button').addEventListener('click', () => {
        state.animationState.playing = true;
        updateAnimationStatus();
    });

    document.getElementById('pause-button').addEventListener('click', () => {
        state.animationState.playing = false;
        updateAnimationStatus();
    });

    document.getElementById('reset-button').addEventListener('click', () => {
        state.animationState.timePhase = 0;
        state.animationState.playing = true;
        updateAnimationStatus();
    });

    // Axis selections
    document.getElementById('x-axis-select').addEventListener('change', updateAllVisualizations);
    document.getElementById('y-axis-select').addEventListener('change', updateAllVisualizations);
}

function updateAnimationStatus() {
    const status = document.getElementById('animation-status');
    status.textContent = state.animationState.playing ? 'Animation playing' : 'Animation paused';
}

function startAnimation() {
    function animate() {
        if (state.animationState.playing) {
            state.animationState.timePhase = (state.animationState.timePhase + state.animationState.speed * 0.2) % (2 * Math.PI);
            updateMotionSilhouette();
            updateGaitCycleAnalysis();
        }
        state.animationFrameId = requestAnimationFrame(animate);
    }
    animate();
}

function updateAllVisualizations() {
    updateMainCorrelationPlot();
    updateMotionSilhouette();
    updateBilateralAsymmetryMotion();
    updateGaitCycleAnalysis();
    updateMotionQualityAssessment();
}

function updateMainCorrelationPlot() {
    const xFeature = document.getElementById('x-axis-select').value;
    const yFeature = document.getElementById('y-axis-select').value;

    const validData = state.dataLoader.getValidDataForFeatures(xFeature, yFeature);

    if (validData.length === 0) {
        Plotly.newPlot('main-correlation-plot', [], {
            title: 'No valid data for selected features',
            template: 'plotly_white'
        }, plotlyConfig);
        return;
    }

    // Group by cohort
    const cohorts = {};
    validData.forEach(row => {
        const cohort = row.COHORT_NAME || 'Unknown';
        if (!cohorts[cohort]) {
            cohorts[cohort] = { x: [], y: [], patno: [] };
        }
        cohorts[cohort].x.push(row[xFeature]);
        cohorts[cohort].y.push(row[yFeature]);
        cohorts[cohort].patno.push(row.PATNO);
    });

    const traces = Object.entries(cohorts).map(([cohort, data]) => ({
        x: data.x,
        y: data.y,
        mode: 'markers',
        type: 'scatter',
        name: cohort,
        marker: {
            size: 8,
            color: COHORT_COLORS[cohort] || '#95a5a6'
        },
        text: data.patno.map(p => `Patient ${p}`),
        hovertemplate: '%{text}<br>%{xaxis.title.text}: %{x:.2f}<br>%{yaxis.title.text}: %{y:.2f}<extra></extra>'
    }));

    // Add selected patient marker
    if (state.selectedPatient && state.currentPatientData) {
        const xVal = state.currentPatientData[xFeature];
        const yVal = state.currentPatientData[yFeature];

        if (xVal !== null && yVal !== null && !isNaN(xVal) && !isNaN(yVal)) {
            traces.push({
                x: [xVal],
                y: [yVal],
                mode: 'markers',
                type: 'scatter',
                name: `Patient ${state.selectedPatient}`,
                marker: {
                    size: 20,
                    color: 'red',
                    symbol: 'star',
                    line: { width: 3, color: 'black' }
                },
                hovertemplate: `Patient ${state.selectedPatient}<br>%{xaxis.title.text}: %{x:.2f}<br>%{yaxis.title.text}: %{y:.2f}<extra></extra>`
            });
        }
    }

    const layout = {
        title: `Analysis: ${FEATURE_LABELS[yFeature]} vs. ${FEATURE_LABELS[xFeature]}`,
        xaxis: { title: FEATURE_LABELS[xFeature] || xFeature },
        yaxis: { title: FEATURE_LABELS[yFeature] || yFeature },
        template: 'plotly_white',
        hovermode: 'closest',
        showlegend: true,
        margin: { l: 60, r: 30, t: 50, b: 60 }
    };

    Plotly.newPlot('main-correlation-plot', traces, layout, plotlyConfig);
}

function updateMotionSilhouette() {
    const motionTest = document.getElementById('motion-test-select').value;
    const timePhase = state.animationState.timePhase;

    const patientData = state.currentPatientData || state.dataLoader.getAveragePatientData();
    const silhouette = state.motionGenerator.generateMotionFrame(patientData, motionTest, timePhase);
    const bodyColors = state.motionGenerator.getBodyColors();

    const traces = [];
    for (const [partName, coords] of Object.entries(silhouette)) {
        const closedX = [...coords.x, coords.x[0]];
        const closedY = [...coords.y, coords.y[0]];

        traces.push({
            x: closedX,
            y: closedY,
            fill: 'toself',
            fillcolor: bodyColors[partName] || '#95a5a6',
            line: { color: 'black', width: 1 },
            mode: 'lines',
            type: 'scatter',
            name: partName.replace(/_/g, ' '),
            showlegend: false,
            hoverinfo: 'name'
        });
    }

    const layout = {
        title: `Motion: ${motionTest.charAt(0).toUpperCase() + motionTest.slice(1)} (Phase: ${timePhase.toFixed(2)})`,
        xaxis: {
            range: [-3, 3],
            showgrid: false,
            zeroline: false,
            showticklabels: false,
            scaleanchor: 'y',
            scaleratio: 1
        },
        yaxis: {
            range: [-4, 9],
            showgrid: false,
            zeroline: false,
            showticklabels: false
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        margin: { l: 10, r: 10, t: 50, b: 10 },
        height: 450
    };

    Plotly.react('motion-silhouette-plot', traces, layout, plotlyConfig);

    // Update metrics display
    updateMotionMetrics(patientData, motionTest);
}

function updateMotionMetrics(patientData, motionTest) {
    const metricsDiv = document.getElementById('motion-metrics-display');

    if (!patientData || Object.keys(patientData).length === 0) {
        metricsDiv.innerHTML = '<p style="color: #7f8c8d; font-style: italic;">No patient selected</p>';
        return;
    }

    let html = '';

    // Speed metric
    const speed = patientData.SP_U;
    if (speed !== null && speed !== undefined && !isNaN(speed)) {
        const speedClass = speed < 0.8 ? 'alert' : speed > 1.2 ? 'good' : 'warning';
        html += `<div class="metric-item">
            <span class="metric-label">Speed:</span>
            <span class="metric-value ${speedClass}">${speed.toFixed(2)} m/s</span>
        </div>`;
    }

    // Asymmetry metric
    const asa = patientData.ASA_U;
    if (asa !== null && asa !== undefined && !isNaN(asa)) {
        const asaClass = asa < 0.2 ? 'good' : asa < 0.5 ? 'warning' : 'alert';
        html += `<div class="metric-item">
            <span class="metric-label">Asymmetry:</span>
            <span class="metric-value ${asaClass}">${asa.toFixed(3)}</span>
        </div>`;
    }

    metricsDiv.innerHTML = html || '<p style="color: #f39c12; font-style: italic;">Limited motion data available</p>';
}

function updateBilateralAsymmetryMotion() {
    const validData = state.dataLoader.getValidDataForFeatures('RA_AMP_U', 'LA_AMP_U');

    if (validData.length === 0) {
        Plotly.newPlot('bilateral-asymmetry-motion', [], {
            title: 'Bilateral arm data not available',
            template: 'plotly_white'
        }, plotlyConfig);
        return;
    }

    const cohorts = {};
    validData.forEach(row => {
        const cohort = row.COHORT_NAME || 'Unknown';
        if (!cohorts[cohort]) {
            cohorts[cohort] = { x: [], y: [] };
        }
        cohorts[cohort].x.push(row.RA_AMP_U);
        cohorts[cohort].y.push(row.LA_AMP_U);
    });

    const traces = Object.entries(cohorts).map(([cohort, data]) => ({
        x: data.x,
        y: data.y,
        mode: 'markers',
        type: 'scatter',
        name: cohort,
        marker: {
            size: 8,
            color: COHORT_COLORS[cohort] || '#95a5a6'
        }
    }));

    // Add diagonal reference line
    const maxVal = Math.max(...validData.map(r => Math.max(r.RA_AMP_U, r.LA_AMP_U)));
    traces.push({
        x: [0, maxVal],
        y: [0, maxVal],
        mode: 'lines',
        type: 'scatter',
        name: 'Perfect Symmetry',
        line: { dash: 'dash', color: 'gray', width: 2 },
        showlegend: false
    });

    // Add selected patient
    if (state.selectedPatient && state.currentPatientData) {
        const raAmp = state.currentPatientData.RA_AMP_U;
        const laAmp = state.currentPatientData.LA_AMP_U;

        if (raAmp !== null && laAmp !== null && !isNaN(raAmp) && !isNaN(laAmp)) {
            traces.push({
                x: [raAmp],
                y: [laAmp],
                mode: 'markers',
                type: 'scatter',
                name: `Patient ${state.selectedPatient}`,
                marker: {
                    size: 20,
                    color: 'red',
                    symbol: 'star',
                    line: { width: 3, color: 'black' }
                }
            });
        }
    }

    const layout = {
        title: 'Bilateral Arm Movement Asymmetry',
        xaxis: { title: 'Right Arm Amplitude' },
        yaxis: { title: 'Left Arm Amplitude' },
        template: 'plotly_white',
        showlegend: true,
        margin: { l: 60, r: 30, t: 50, b: 60 }
    };

    Plotly.newPlot('bilateral-asymmetry-motion', traces, layout, plotlyConfig);
}

function updateGaitCycleAnalysis() {
    if (!state.selectedPatient || !state.currentPatientData) {
        Plotly.newPlot('gait-cycle-analysis', [], {
            title: 'Select a patient to see gait cycle',
            template: 'plotly_white'
        }, plotlyConfig);
        return;
    }

    const leftAmp = state.currentPatientData.LA_AMP_U || 30;
    const rightAmp = state.currentPatientData.RA_AMP_U || 30;

    const timePoints = [];
    const leftSwing = [];
    const rightSwing = [];

    for (let i = 0; i <= 100; i++) {
        const t = (i / 100) * 2 * Math.PI;
        timePoints.push(t);
        leftSwing.push((leftAmp / 50.0) * Math.sin(t));
        rightSwing.push((rightAmp / 50.0) * Math.sin(t + Math.PI));
    }

    const traces = [
        {
            x: timePoints,
            y: leftSwing,
            mode: 'lines',
            type: 'scatter',
            name: 'Left Arm Swing',
            line: { color: '#e74c3c', width: 3 }
        },
        {
            x: timePoints,
            y: rightSwing,
            mode: 'lines',
            type: 'scatter',
            name: 'Right Arm Swing',
            line: { color: '#27ae60', width: 3 }
        },
        {
            x: [state.animationState.timePhase, state.animationState.timePhase],
            y: [-1, 1],
            mode: 'lines',
            type: 'scatter',
            name: 'Current Phase',
            line: { dash: 'dash', color: 'red', width: 2 }
        }
    ];

    const layout = {
        title: `Gait Cycle - Patient ${state.selectedPatient}`,
        xaxis: { title: 'Phase (radians)' },
        yaxis: { title: 'Amplitude' },
        template: 'plotly_white',
        showlegend: true,
        margin: { l: 60, r: 30, t: 50, b: 60 }
    };

    Plotly.react('gait-cycle-analysis', traces, layout, plotlyConfig);
}

function updateMotionQualityAssessment() {
    if (!state.selectedPatient || !state.currentPatientData) {
        Plotly.newPlot('motion-quality-assessment', [], {
            title: 'Select a patient to see quality assessment',
            template: 'plotly_white'
        }, plotlyConfig);
        return;
    }

    const movementQuality = Math.min((state.currentPatientData.MOVEMENT_QUALITY || 0) / 20, 1.0);
    const coordination = state.currentPatientData.BILATERAL_COORDINATION || 0;
    const symmetry = Math.max(0, 1 - ((state.currentPatientData.ASA_U || 2.0) / 2.0));

    const trace = {
        r: [movementQuality, coordination, symmetry],
        theta: ['Movement Quality', 'Coordination', 'Symmetry'],
        fill: 'toself',
        type: 'scatterpolar',
        name: 'Patient Quality',
        marker: { color: '#3498db' },
        line: { color: '#2980b9', width: 2 }
    };

    const layout = {
        title: `Motion Quality - Patient ${state.selectedPatient}`,
        polar: {
            radialaxis: {
                visible: true,
                range: [0, 1]
            }
        },
        template: 'plotly_white',
        showlegend: false,
        margin: { l: 60, r: 60, t: 50, b: 60 }
    };

    Plotly.newPlot('motion-quality-assessment', [trace], layout, plotlyConfig);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
