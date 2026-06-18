/**
 * app.js — Orchestration: data + 3D figure + motion engine + charts + controls.
 */

import { ParkinsonDataLoader, FEATURE_LABELS, COHORT_COLORS } from './data-loader.js';
import { Figure3D } from './figure3d.js';
import { computePose, gaitPhase, affectedArm } from './motion.js';
import {
    correlationPlot, bilateralPlot, gaitCyclePlot, updateGaitPhase, qualityRadar,
} from './charts.js';

const state = {
    loader: null,
    figure: null,
    patient: null,          // current participant data (or cohort average)
    motionType: 'gait',
    speed: 1.0,
    playing: true,
    clock: 0,               // seconds of motion time (scaled by speed)
    phaseAccum: 0,          // throttle for gait-phase chart updates
};

const $ = (id) => document.getElementById(id);

async function init() {
    state.loader = new ParkinsonDataLoader();
    try {
        await state.loader.loadAllDatasets();
    } catch (err) {
        $('figure-3d').innerHTML = `<div class="figure-3d__loading">Could not load data.<br><small>${err.message}</small></div>`;
        return;
    }

    state.patient = state.loader.getAveragePatientData();

    try {
        state.figure = new Figure3D($('figure-3d'), { onFrame });
        state.figure.start();
        const loading = $('figure-3d-loading');
        if (loading) loading.remove();
    } catch (err) {
        $('figure-3d').innerHTML = `<div class="figure-3d__loading">3D model failed to load.<br><small>${err.message}</small></div>`;
    }

    populatePatients();
    wireControls();
    applyPatient();          // summary, metrics, charts, accent
    drawAllCharts();
}

// ── Animation frame (driven by Figure3D's render loop) ──────────────────────
function onFrame(dt) {
    if (state.playing) state.clock += dt * state.speed;

    if (state.figure) {
        const pose = computePose(state.patient, state.motionType, state.clock);
        state.figure.applyPose(pose);
    }

    // Caption + throttled gait-phase marker.
    if (state.motionType === 'gait' || state.motionType === 'tug') {
        const ph = gaitPhase(state.patient, state.clock);
        $('figure-caption').textContent = captionFor(state.motionType, ph);
        state.phaseAccum += dt;
        if (state.phaseAccum > 0.05) {
            state.phaseAccum = 0;
            updateGaitPhase($('gait-cycle-analysis'), ph);
        }
    } else {
        $('figure-caption').textContent = captionFor(state.motionType);
    }
}

function captionFor(type, ph) {
    switch (type) {
        case 'gait': return `Gait / walking · cycle ${(ph / (2 * Math.PI)).toFixed(2)}`;
        case 'tug': return 'Timed Up & Go';
        case 'balance': return 'Postural sway / balance';
        case 'free': return 'Free / idle';
        default: return '';
    }
}

// ── Controls ────────────────────────────────────────────────────────────────
function populatePatients() {
    const sel = $('patient-select');
    state.loader.getPatients().forEach(p => {
        const o = document.createElement('option');
        o.value = p; o.textContent = `Participant ${p}`;
        sel.appendChild(o);
    });
}

function wireControls() {
    $('patient-select').addEventListener('change', (e) => {
        const v = e.target.value;
        state.patient = v ? state.loader.getPatientData(parseInt(v, 10)) : state.loader.getAveragePatientData();
        applyPatient();
        drawAllCharts();
    });

    $('motion-test-select').addEventListener('change', (e) => {
        state.motionType = e.target.value;
        state.clock = 0;
    });

    const speed = $('animation-speed');
    speed.addEventListener('input', (e) => {
        state.speed = parseFloat(e.target.value);
        $('speed-display').textContent = `${state.speed.toFixed(1)}×`;
    });

    $('play-button').addEventListener('click', () => setPlaying(true));
    $('pause-button').addEventListener('click', () => setPlaying(false));
    $('reset-button').addEventListener('click', () => { state.clock = 0; setPlaying(true); });

    document.querySelectorAll('.view-presets .chip').forEach(btn => {
        btn.addEventListener('click', () => {
            const view = btn.dataset.view;
            state.figure && state.figure.setView(view);
            document.querySelectorAll('.view-presets .chip').forEach(b => b.setAttribute('aria-pressed', 'false'));
            if (view !== 'reset') btn.setAttribute('aria-pressed', 'true');
        });
    });

    $('x-axis-select').addEventListener('change', drawCorrelation);
    $('y-axis-select').addEventListener('change', drawCorrelation);
}

function setPlaying(on) {
    state.playing = on;
    const pill = $('animation-status');
    pill.textContent = on ? 'Playing' : 'Paused';
    pill.classList.toggle('is-playing', on);
    pill.classList.toggle('is-paused', !on);
    $('play-button').setAttribute('aria-pressed', String(on));
    $('pause-button').setAttribute('aria-pressed', String(!on));
}

// ── Participant summary + metrics + figure accent ───────────────────────────
function applyPatient() {
    const p = state.patient || {};
    const isAvg = p.PATNO == null;
    const cohort = p.COHORT_NAME || 'Unknown';

    const badge = $('cohort-badge');
    badge.textContent = isAvg ? 'Cohort average' : cohort;
    badge.style.setProperty('--badge', COHORT_COLORS[cohort] || COHORT_COLORS.Unknown);

    $('patient-summary').innerHTML = summaryRows(p, isAvg);
    $('motion-metrics-display').innerHTML = metricChips(p);

    // Mark the more-affected arm on the model (subtle amber tint).
    if (state.figure) {
        if (isAvg) state.figure.setAffected(null, 0);
        else {
            const a = affectedArm(p);
            state.figure.setAffected(a.amount > 0.12 ? a.side : null, 0.25 + 0.4 * a.amount);
        }
    }
}

function summaryRows(p, isAvg) {
    const rows = [
        ['Cohort', isAvg ? 'Average across cohort' : (p.COHORT_NAME || '—')],
        ['Age', isAvg ? '—' : fmt(p.ENROLL_AGE, 0)],
        ['Sex', isAvg ? '—' : sexLabel(p.SEX)],
        ['Handedness', isAvg ? '—' : handedLabel(p.HANDED)],
        ['Hoehn–Yahr stage', isAvg ? '—' : clinical(p.NHY, 0)],
        ['UPDRS-III (motor)', isAvg ? fmt(p.NP3TOT, 0) : clinical(p.NP3TOT, 0)],
        ['UPDRS-II (patient)', isAvg ? fmt(p.NP2PTOT, 0) : clinical(p.NP2PTOT, 0)],
    ];
    return rows.map(([k, v]) => `
        <div class="summary-row"><span class="summary-key">${k}</span><span class="summary-val">${v}</span></div>
    `).join('');
}

function metricChips(p) {
    const speed = numOr(p.SP_U);
    const asym = numOr(p.ASA_U);
    const dtc = numOr(p.DUAL_TASK_COST);
    const cad = numOr(p.CAD_U);
    const trR = numOr(p.NP3PTRMR), trL = numOr(p.NP3PTRML);
    const hasTremorExam = trR != null || trL != null;
    const tremor = (trR || 0) + (trL || 0);

    const chips = [];
    if (speed != null) chips.push(chip('Gait speed', `${speed.toFixed(2)} m/s`,
        speed < 0.8 ? 'alert' : speed < 1.1 ? 'warn' : 'good'));
    if (cad != null) chips.push(chip('Cadence', `${cad.toFixed(0)} /min`, 'neutral'));
    if (asym != null) chips.push(chip('Arm-swing asymmetry', asym.toFixed(1),
        asym < 10 ? 'good' : asym < 25 ? 'warn' : 'alert'));
    if (dtc != null) chips.push(chip('Dual-task cost', `${dtc.toFixed(1)} %`,
        dtc < 5 ? 'good' : dtc < 15 ? 'warn' : 'alert'));
    if (hasTremorExam) chips.push(chip('Rest/postural tremor', tremor.toFixed(0),
        tremor === 0 ? 'good' : tremor < 3 ? 'warn' : 'alert'));

    return chips.join('') || '<p class="muted">No movement metrics available.</p>';
}

function chip(label, value, status) {
    return `<div class="metric ${status}">
        <span class="metric__label">${label}</span>
        <span class="metric__value">${value}</span>
    </div>`;
}

// ── Charts ──────────────────────────────────────────────────────────────────
function drawAllCharts() {
    drawCorrelation();
    bilateralPlot($('bilateral-asymmetry-motion'),
        state.loader.getValidDataForFeatures('RA_AMP_U', 'LA_AMP_U'), state.patient);
    gaitCyclePlot($('gait-cycle-analysis'), state.patient, gaitPhase(state.patient, state.clock));
    qualityRadar($('motion-quality-assessment'), state.patient);
}

function drawCorrelation() {
    const xF = $('x-axis-select').value;
    const yF = $('y-axis-select').value;
    correlationPlot($('main-correlation-plot'),
        state.loader.getValidDataForFeatures(xF, yF), xF, yF, state.patient);
}

// ── formatting helpers ──────────────────────────────────────────────────────
function numOr(v) { return (v === null || v === undefined || isNaN(v)) ? null : Number(v); }
function fmt(v, d = 1) { const n = numOr(v); return n == null ? '—' : n.toFixed(d); }
function clinical(v, d = 0) { return numOr(v) == null ? 'Unknown' : fmt(v, d); }
function sexLabel(v) { return v === 0 || v === '0' ? 'Female' : v === 1 || v === '1' ? 'Male' : '—'; }
function handedLabel(v) {
    return ({ 1: 'Right', 2: 'Left', 3: 'Mixed' })[v] || '—';
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
