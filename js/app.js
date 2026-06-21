/**
 * app.js — Orchestration: data + 3D figure + motion engine + charts + controls.
 */

import { ParkinsonDataLoader, FEATURE_LABELS, COHORT_COLORS } from './data-loader.js';
import { Figure3D } from './figure3d.js';
import { computePose, gaitPhase, affectedArm } from './motion.js';
import {
    correlationPlot, bilateralPlot, gaitCyclePlot, updateGaitPhase, qualityRadar, realWaveform,
} from './charts.js';

const state = {
    loader: null,
    figure: null,
    patient: null,          // current participant data (or cohort average)
    motionType: 'gait',
    speed: 1.0,
    playing: true,
    clock: 0,               // seconds of motion time (scaled by speed)
    wallClock: 0,           // unscaled real seconds (tremor freq must not track the speed slider)
    phaseAccum: 0,          // throttle for gait-phase chart updates
    clip: null,             // loaded real-motion clip (WearGait)
    clipLoading: false,
    clipError: false,       // last WearGait clip fetch failed
    clipReq: 0,             // request token to ignore superseded async clip loads
    manifest: null,         // available real-motion clips
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
    updateModeTag();

    // Respect reduced-motion preference: start paused (the user can press Play).
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        setPlaying(false);
    }
}

// ── Animation frame (driven by Figure3D's render loop) ──────────────────────
function onFrame(dt) {
    if (state.playing) { state.clock += dt * state.speed; state.wallClock += dt; }

    const wgClip = (state.motionType === 'weargait' && state.clip) ? state.clip : null;
    let wgIdx = 0;
    if (state.figure) {
        if (wgClip) {
            wgIdx = Math.floor(state.clock * wgClip.fps) % wgClip.frames.length;
            state.figure.applyPose(wgClip.frames[wgIdx]);
        } else {
            // No clip yet (loading/failed) or non-WearGait mode. computePose returns a
            // neutral idle pose for 'weargait', so we never show synthetic gait under the IMU tag.
            state.figure.applyPose(computePose(state.patient, state.motionType, state.clock, state.wallClock));
        }
    }

    // Caption + throttled phase marker on the waveform chart.
    if (state.motionType === 'weargait') {
        $('figure-caption').textContent = wgClip
            ? `WearGait · ${wgClip.id} · ${wgClip.asymmetryPct}% arm-swing asym (IMU-derived)`
            : (state.clipError ? 'IMU clip unavailable' : 'Loading IMU clip…');
        if (wgClip) {
            state.phaseAccum += dt;
            if (state.phaseAccum > 0.05) {
                state.phaseAccum = 0;
                updateGaitPhase($('gait-cycle-analysis'), wgIdx / (wgClip.frames.length - 1));
            }
        }
    } else if (state.motionType === 'gait' || state.motionType === 'tug') {
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

async function loadManifest() {
    if (state.manifest) return;
    const r = await fetch('data/motion_clips/index.json');
    state.manifest = (await r.json()).clips;
    const sel = $('weargait-select');
    sel.innerHTML = '';
    state.manifest.forEach(c => {
        const o = document.createElement('option');
        o.value = c.id;
        o.textContent = `${c.id} — ${c.cohort} · ${c.asymmetryPct}% asym`;
        sel.appendChild(o);
    });
}

async function loadClip(id) {
    const req = ++state.clipReq;
    let data = null;
    try {
        const r = await fetch('data/motion_clips/' + encodeURIComponent(id) + '.json');
        if (r.ok) data = await r.json();
    } catch (e) {
        console.error('clip load failed:', e);
    }
    // Ignore if a newer clip/mode selection superseded this fetch.
    if (req !== state.clipReq || state.motionType !== 'weargait') return;
    if (data) { state.clip = data; state.clipError = false; applyWearGait(state.clip); updateModeTag(); }
    else { state.clipError = true; updateModeTag(); }
}

async function enterWearGait() {
    if (state.clipLoading) return;
    state.clipLoading = true;
    try {
        await loadManifest();
        const sel = $('weargait-select');
        const id = sel.value || (state.manifest[0] && state.manifest[0].id);
        if (id) { sel.value = id; await loadClip(id); }
    } catch (e) {
        console.error('WearGait load failed:', e);
    }
    state.clipLoading = false;
}

// ── Controls ────────────────────────────────────────────────────────────────
function populatePatients() {
    const sel = $('patient-select');
    state.loader.getPatients().forEach(p => {
        const o = document.createElement('option');
        o.value = p; o.textContent = `Participant ${p}`;
        sel.appendChild(o);
    });
    const rc = $('record-count');
    if (rc) rc.textContent = state.loader.records.length;
}

function wireControls() {
    $('patient-select').addEventListener('change', (e) => {
        const v = e.target.value;
        state.patient = v ? state.loader.getPatientData(parseInt(v, 10)) : state.loader.getAveragePatientData();
        if (state.motionType !== 'weargait') applyPatient();   // WearGait owns the readout while active
        drawAllCharts();
    });

    $('motion-test-select').addEventListener('change', (e) => {
        state.motionType = e.target.value;
        state.clock = 0;
        state.clipError = false;
        updateModeTag();
        const wg = state.motionType === 'weargait';
        $('weargait-control').style.display = wg ? '' : 'none';
        if (wg) { drawAllCharts(); enterWearGait(); }   // drop PPMI highlight now; clip loads async
        else { applyPatient(); drawAllCharts(); }       // restore PPMI readout + charts
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

    $('weargait-select').addEventListener('change', async (e) => {
        state.clock = 0;
        await loadClip(e.target.value);
    });
}

function updateModeTag() {
    const tag = $('model-mode-tag');
    if (!tag) return;
    if (state.motionType === 'weargait') {
        if (state.clip) { tag.textContent = 'IMU-derived · Synapse WearGait'; tag.className = 'model-tag measured'; }
        else if (state.clipError) { tag.textContent = 'IMU clip unavailable'; tag.className = 'model-tag schematic'; }
        else { tag.textContent = 'Loading IMU clip…'; tag.className = 'model-tag schematic'; }
    } else {
        tag.textContent = 'Schematic · modeled from PPMI metrics';
        tag.className = 'model-tag schematic';
    }
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

function applyWearGait(clip) {
    const cohortName = clip.cohort === 'PD' ? "Parkinson's Disease" : 'Healthy Control';
    const badge = $('cohort-badge');
    badge.textContent = `WearGait · ${clip.cohort}`;
    badge.style.setProperty('--badge', COHORT_COLORS[cohortName] || COHORT_COLORS.Unknown);

    const rows = [
        ['Source', 'Synapse WearGait (IMU-derived)'],
        ['Participant', clip.id],
        ['Cohort', cohortName],
        ['Age', clip.age == null ? '—' : String(clip.age)],
        ['Sex', clip.sex || '—'],
        ['Hoehn–Yahr stage', clip.hy == null ? '—' : String(clip.hy)],
        ['UPDRS-III (motor)', clip.updrs3 == null ? '—' : String(clip.updrs3)],
        ['Provenance', 'Gait timing &amp; arm-swing measured; knees / trunk / joint angles estimated'],
    ];
    $('patient-summary').innerHTML = rows.map(([k, v]) =>
        `<div class="summary-row"><span class="summary-key">${k}</span><span class="summary-val">${v}</span></div>`).join('');

    const lowQ = clip.armQuality !== 'ok';
    const chips = [];
    chips.push(chip(`Arm-swing asymmetry${lowQ ? ' (low signal)' : ''}`, `${clip.asymmetryPct}%`,
        lowQ ? 'neutral' : clip.asymmetryPct < 15 ? 'good' : clip.asymmetryPct < 35 ? 'warn' : 'alert'));
    chips.push(chip('Cadence', `${Math.round(clip.gaitHz * 120)} /min`, 'neutral'));
    if (clip.updrs3 != null) chips.push(chip('UPDRS-III', String(clip.updrs3),
        clip.updrs3 < 20 ? 'good' : clip.updrs3 < 40 ? 'warn' : 'alert'));
    $('motion-metrics-display').innerHTML = chips.join('');
    drawAllCharts();   // redraw all charts WearGait-aware (measured waveform; no PPMI highlight)

    if (state.figure) {
        const aL = clip.armAmtL, aR = clip.armAmtR;
        const amt = Math.abs(aL - aR) / (aL + aR + 1e-6);
        state.figure.setAffected(!lowQ && amt > 0.12 ? (aL < aR ? 'l' : 'r') : null, 0.25 + 0.4 * amt);
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
        ['Medication state', isAvg ? '—' : medState(p.PDSTATE)],
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
    if (hasTremorExam) chips.push(chip('Postural tremor (UPDRS 3.15)', tremor.toFixed(0),
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
    const wg = state.motionType === 'weargait';
    const sel = wg ? null : state.patient;   // don't tie a PPMI participant to a WearGait clip
    drawCorrelation();
    bilateralPlot($('bilateral-asymmetry-motion'),
        state.loader.getValidDataForFeatures('RA_AMP_U', 'LA_AMP_U'), sel);
    if (wg && state.clip) {
        realWaveform($('gait-cycle-analysis'), state.clip, 0);   // measured WearGait waveform
    } else {
        gaitCyclePlot($('gait-cycle-analysis'), state.patient, gaitPhase(state.patient, state.clock));
    }
    qualityRadar($('motion-quality-assessment'), sel);   // radar has no WearGait analogue → empty there
}

function drawCorrelation() {
    const xF = $('x-axis-select').value;
    const yF = $('y-axis-select').value;
    const sel = state.motionType === 'weargait' ? null : state.patient;
    correlationPlot($('main-correlation-plot'),
        state.loader.getValidDataForFeatures(xF, yF), xF, yF, sel);
}

// ── formatting helpers ──────────────────────────────────────────────────────
function numOr(v) { return (v === null || v === undefined || isNaN(v)) ? null : Number(v); }
function fmt(v, d = 1) { const n = numOr(v); return n == null ? '—' : n.toFixed(d); }
function clinical(v, d = 0) { return numOr(v) == null ? 'Unknown' : fmt(v, d); }
function sexLabel(v) { return v === 0 || v === '0' ? 'Female' : v === 1 || v === '1' ? 'Male' : '—'; }
function handedLabel(v) {
    return ({ 1: 'Right', 2: 'Left', 3: 'Mixed' })[v] || '—';
}
function medState(v) {
    const s = String(v == null ? '' : v).toUpperCase();
    return s === 'OFF' ? 'OFF (unmedicated)' : s === 'ON' ? 'ON (medicated)' : 'Unknown';
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
