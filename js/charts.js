/**
 * charts.js — Plotly 2D analysis charts with a restrained, professional theme.
 */

import { FEATURE_LABELS, COHORT_COLORS } from './data-loader.js';

const FONT = 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
const INK = '#23272e';
const MUTED = '#5b6470';
const GRID = '#e9ecf1';
const AXIS = '#d4d9e0';

export const PLOTLY_CONFIG = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
};

function theme(extra = {}) {
    return Object.assign({
        font: { family: FONT, size: 12, color: MUTED },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        title: { font: { size: 14, color: INK }, x: 0.01, xanchor: 'left', y: 0.97 },
        margin: { l: 56, r: 18, t: 44, b: 48 },
        hovermode: 'closest',
        legend: {
            orientation: 'h', x: 0, y: 1.12, xanchor: 'left',
            font: { size: 11, color: MUTED }, bgcolor: 'rgba(0,0,0,0)',
        },
        colorway: ['#4a6fa5', '#3f7d6e', '#c2913e', '#c2533f', '#6b5b95'],
    }, extra);
}

function axis(title) {
    return {
        title: { text: title, font: { size: 12, color: MUTED }, standoff: 8 },
        gridcolor: GRID, zeroline: false, linecolor: AXIS,
        ticks: 'outside', tickcolor: AXIS, tickfont: { size: 11 },
    };
}

function highlightTrace(x, y, label) {
    return {
        x: [x], y: [y], mode: 'markers', type: 'scatter', name: label,
        marker: { symbol: 'circle-open', size: 17, color: '#16191d', line: { width: 3, color: '#16191d' } },
        hovertemplate: `${label}<extra></extra>`, showlegend: false,
    };
}

function emptyPlot(container, message) {
    Plotly.react(container, [], theme({
        annotations: [{
            text: message, showarrow: false,
            font: { size: 13, color: MUTED }, xref: 'paper', yref: 'paper', x: 0.5, y: 0.5,
        }],
        xaxis: { visible: false }, yaxis: { visible: false },
    }), PLOTLY_CONFIG);
}

// ── Main correlation scatter ────────────────────────────────────────────────
export function correlationPlot(container, validData, xF, yF, selected) {
    if (!validData.length) return emptyPlot(container, 'No data for the selected features');

    const cohorts = groupByCohort(validData, xF, yF);
    const traces = Object.entries(cohorts).map(([cohort, d]) => ({
        x: d.x, y: d.y, mode: 'markers', type: 'scatter', name: cohort,
        marker: { size: 7, color: COHORT_COLORS[cohort] || COHORT_COLORS.Unknown, opacity: 0.82,
                  line: { width: 0.5, color: 'rgba(255,255,255,0.6)' } },
        text: d.patno.map(p => `Participant ${p}`),
        hovertemplate: '%{text}<br>%{x:.2f}, %{y:.2f}<extra></extra>',
    }));

    if (selected && selected.PATNO != null && isNum(selected[xF]) && isNum(selected[yF])) {
        traces.push(highlightTrace(selected[xF], selected[yF], `Participant ${selected.PATNO}`));
    }

    Plotly.react(container, traces, theme({
        title: `${FEATURE_LABELS[yF] || yF} vs. ${FEATURE_LABELS[xF] || xF}`,
        xaxis: axis(FEATURE_LABELS[xF] || xF),
        yaxis: axis(FEATURE_LABELS[yF] || yF),
    }), PLOTLY_CONFIG);
}

// ── Bilateral arm-swing asymmetry ──────────────────────────────────────────
export function bilateralPlot(container, validData, selected) {
    if (!validData.length) return emptyPlot(container, 'Bilateral arm data unavailable');

    const cohorts = groupByCohort(validData, 'RA_AMP_U', 'LA_AMP_U');
    const traces = Object.entries(cohorts).map(([cohort, d]) => ({
        x: d.x, y: d.y, mode: 'markers', type: 'scatter', name: cohort,
        marker: { size: 7, color: COHORT_COLORS[cohort] || COHORT_COLORS.Unknown, opacity: 0.82 },
        showlegend: false,
    }));

    const maxV = Math.max(...validData.map(r => Math.max(r.RA_AMP_U, r.LA_AMP_U))) * 1.05;
    traces.push({
        x: [0, maxV], y: [0, maxV], mode: 'lines', type: 'scatter', name: 'Symmetry',
        line: { dash: 'dot', color: '#aab2bd', width: 1.5 }, hoverinfo: 'skip', showlegend: false,
    });

    if (selected && selected.PATNO != null && isNum(selected.RA_AMP_U) && isNum(selected.LA_AMP_U)) {
        traces.push(highlightTrace(selected.RA_AMP_U, selected.LA_AMP_U, `Participant ${selected.PATNO}`));
    }

    Plotly.react(container, traces, theme({
        title: 'Bilateral arm-swing amplitude (R vs L)',
        xaxis: axis('Right arm amplitude (°)'),
        yaxis: axis('Left arm amplitude (°)'),
    }), PLOTLY_CONFIG);
}

// ── Gait cycle: arm-swing waveforms + live phase marker ────────────────────
export function gaitCyclePlot(container, patient, phase) {
    if (!patient) return emptyPlot(container, 'Select a participant');
    const la = num(patient.LA_AMP_U, 26), ra = num(patient.RA_AMP_U, 26);
    const N = 90, x = [], left = [], right = [];
    for (let i = 0; i <= N; i++) {
        const t = (i / N) * 2 * Math.PI;
        x.push(t);
        left.push((la / 2) * Math.sin(t));
        right.push((ra / 2) * Math.sin(t + Math.PI));
    }
    const traces = [
        { x, y: left, mode: 'lines', name: 'Left arm', line: { color: '#4a6fa5', width: 2.5 } },
        { x, y: right, mode: 'lines', name: 'Right arm', line: { color: '#c2533f', width: 2.5 } },
    ];
    Plotly.react(container, traces, theme({
        title: 'Arm-swing waveform over gait cycle',
        xaxis: Object.assign(axis('Gait-cycle phase'), {
            range: [0, 2 * Math.PI],
            tickvals: [0, Math.PI / 2, Math.PI, 1.5 * Math.PI, 2 * Math.PI],
            ticktext: ['0', '¼', '½', '¾', '1'],
        }),
        yaxis: axis('Arm swing (°)'),
        shapes: [phaseShape(phase)],
    }), PLOTLY_CONFIG);
}

export function updateGaitPhase(container, phase) {
    if (!container || !container.layout) return;
    Plotly.relayout(container, { 'shapes[0].x0': phase, 'shapes[0].x1': phase });
}

function phaseShape(phase) {
    return {
        type: 'line', x0: phase || 0, x1: phase || 0, y0: 0, y1: 1, yref: 'paper',
        line: { color: '#16191d', width: 1.5, dash: 'dot' },
    };
}

// ── Movement-quality radar ─────────────────────────────────────────────────
export function qualityRadar(container, patient) {
    if (!patient) return emptyPlot(container, 'Select a participant');
    const movement = clamp01(num(patient.MOVEMENT_QUALITY, 0) / 20);
    const coordination = clamp01(num(patient.BILATERAL_COORDINATION, 0));
    const symmetry = clamp01(1 - num(patient.ASA_U, 2) / 2);
    const smoothness = clamp01(1 - num(patient.TOTAL_JERK, 0.05) / 0.1);
    const speed = clamp01(num(patient.SP_U, 1) / 1.4);

    const cats = ['Movement\nquality', 'Coordination', 'Symmetry', 'Smoothness', 'Speed'];
    const vals = [movement, coordination, symmetry, smoothness, speed];

    Plotly.react(container, [{
        type: 'scatterpolar', r: [...vals, vals[0]], theta: [...cats, cats[0]],
        fill: 'toself', fillcolor: 'rgba(74,111,165,0.18)',
        line: { color: '#4a6fa5', width: 2 }, name: 'Participant',
    }], theme({
        title: 'Movement-quality profile',
        margin: { l: 56, r: 56, t: 44, b: 36 },
        polar: {
            bgcolor: 'rgba(0,0,0,0)',
            radialaxis: { visible: true, range: [0, 1], gridcolor: GRID, tickfont: { size: 9 }, angle: 90 },
            angularaxis: { gridcolor: GRID, tickfont: { size: 10, color: MUTED } },
        },
        showlegend: false,
    }), PLOTLY_CONFIG);
}

// ── helpers ────────────────────────────────────────────────────────────────
function groupByCohort(rows, xF, yF) {
    const g = {};
    rows.forEach(r => {
        const c = r.COHORT_NAME || 'Unknown';
        (g[c] = g[c] || { x: [], y: [], patno: [] });
        g[c].x.push(r[xF]); g[c].y.push(r[yF]); g[c].patno.push(r.PATNO);
    });
    return g;
}
const isNum = v => v !== null && v !== undefined && !isNaN(v);
const num = (v, d) => (isNum(v) ? Number(v) : d);
const clamp01 = v => Math.max(0, Math.min(1, v));
