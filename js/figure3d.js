/**
 * figure3d.js — Real-time articulated 3D human model (Three.js).
 *
 * The figure is a true forward-kinematics joint hierarchy: every limb segment
 * is a child Group of its proximal joint, so rotating a joint moves the whole
 * chain and the body can never come apart. Motion is applied as joint rotations
 * via applyPose(); see motion.js for how a patient's data becomes a pose.
 *
 * Coordinate frame: Y up, figure faces +Z. Limb swing during gait is rotation
 * about X (the medio-lateral axis), i.e. fore/aft in the sagittal (Y–Z) plane.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Anatomical proportions (metres, ~1.72 m figure) ────────────────────────
const HIP_Y = 0.97;          // pelvis height (root origin); set so the feet rest on the ground (y≈0)
const TORSO_LEN = 0.50;      // pelvis → shoulder line
const SHOULDER_HALF = 0.185; // half shoulder width
const NECK_LEN = 0.095;
const HEAD_R = 0.107;
const UP_ARM = 0.28, FOREARM = 0.255, HAND_LEN = 0.10;
const HIP_HALF = 0.10;
const THIGH = 0.45, SHIN = 0.43, FOOT_LEN = 0.21;

// Limb radii
const R = {
    torso: 0.135, pelvis: 0.115, upArm: 0.052, forearm: 0.044, hand: 0.040,
    thigh: 0.082, shin: 0.062, neck: 0.053,
};

const COLORS = {
    body: 0x9aa6b8,        // cool clinical slate
    accent: 0xd98a3d,      // restrained amber — marks the more-affected side
    background: 0xeef1f5,
};

// ── small geometry helpers ─────────────────────────────────────────────────
function makeBone(parent, length, radius, material, { sx = 1, sz = 1 } = {}) {
    const cyl = Math.max(length - 1.4 * radius, 0.01);
    const mesh = new THREE.Mesh(new THREE.CapsuleGeometry(radius, cyl, 8, 18), material);
    mesh.position.y = -length / 2;       // bone hangs downward from the joint at y=0
    mesh.scale.set(sx, 1, sz);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    parent.add(mesh);
    return mesh;
}

function makeJointBall(parent, radius, material, y = 0) {
    const s = new THREE.Mesh(new THREE.SphereGeometry(radius, 16, 12), material);
    s.position.y = y;
    s.castShadow = true;
    parent.add(s);
    return s;
}

function group(parent, x = 0, y = 0, z = 0) {
    const g = new THREE.Group();
    g.position.set(x, y, z);
    parent.add(g);
    return g;
}

export class Figure3D {
    /**
     * @param {HTMLElement} container
     * @param {{onFrame?: (dtSeconds:number)=>void}} [opts]
     */
    constructor(container, opts = {}) {
        this.container = container;
        this.onFrame = opts.onFrame || null;
        this._raf = null;
        this._lastT = 0;
        this._camTweening = false;
        this._camDest = new THREE.Vector3();

        this._initRenderer();
        this._initScene();
        this._buildFigure();
        this._initControls();
        this.setView('threeQuarter', true);

        this._onResize = this._resize.bind(this);
        this._ro = new ResizeObserver(this._onResize);
        this._ro.observe(container);
        window.addEventListener('resize', this._onResize);
    }

    _initRenderer() {
        const r = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        r.setClearColor(0x000000, 0);
        r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        r.shadowMap.enabled = true;
        r.shadowMap.type = THREE.PCFSoftShadowMap;
        r.toneMapping = THREE.ACESFilmicToneMapping;
        r.toneMappingExposure = 1.05;
        this.renderer = r;
        this.container.appendChild(r.domElement);
        r.domElement.style.display = 'block';
        r.domElement.style.touchAction = 'none';

        // Recover gracefully if the browser drops the WebGL context (tab backgrounding, GPU reset).
        r.domElement.addEventListener('webglcontextlost', (e) => {
            e.preventDefault();
            if (this._raf) { cancelAnimationFrame(this._raf); this._raf = null; }
            const cap = document.getElementById('figure-caption');
            if (cap) cap.textContent = 'Restoring 3D view…';
        }, false);
        r.domElement.addEventListener('webglcontextrestored', () => {
            this._lastT = 0;
            this.start();   // onFrame resumes and overwrites the caption
        }, false);
    }

    _initScene() {
        const scene = new THREE.Scene();
        this.scene = scene;

        const w = this.container.clientWidth || 600;
        const h = this.container.clientHeight || 460;
        this.camera = new THREE.PerspectiveCamera(38, w / h, 0.1, 100);
        this.camera.position.set(2.4, 1.7, 3.0);
        this.renderer.setSize(w, h);

        // Lighting: soft hemisphere ambient + key + fill, for a studio look.
        const hemi = new THREE.HemisphereLight(0xf4f7fb, 0x4a4f59, 1.7);
        scene.add(hemi);

        const key = new THREE.DirectionalLight(0xffffff, 2.6);
        key.position.set(3.2, 6.0, 4.5);
        key.castShadow = true;
        key.shadow.mapSize.set(2048, 2048);
        key.shadow.radius = 6;
        key.shadow.bias = -0.0004;
        const c = key.shadow.camera;
        c.near = 0.5; c.far = 20; c.left = -2; c.right = 2; c.top = 3; c.bottom = -1;
        scene.add(key);

        const fill = new THREE.DirectionalLight(0xdfe6f0, 0.55);
        fill.position.set(-4, 2.5, -2);
        scene.add(fill);

        const rim = new THREE.DirectionalLight(0xbcccdc, 0.5);
        rim.position.set(-1.5, 3.5, -5);
        scene.add(rim);

        // Ground: invisible plane that only catches the contact shadow.
        const ground = new THREE.Mesh(
            new THREE.PlaneGeometry(12, 12),
            new THREE.ShadowMaterial({ opacity: 0.20 })
        );
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = 0;
        ground.receiveShadow = true;
        scene.add(ground);

        // Faint grounding disc for a sense of place (very subtle).
        const disc = new THREE.Mesh(
            new THREE.CircleGeometry(1.15, 48),
            new THREE.MeshBasicMaterial({ color: 0xdfe4ec, transparent: true, opacity: 0.5 })
        );
        disc.rotation.x = -Math.PI / 2;
        disc.position.y = 0.001;
        scene.add(disc);
    }

    _mat(color) {
        return new THREE.MeshStandardMaterial({
            color, roughness: 0.62, metalness: 0.04,
        });
    }

    _buildFigure() {
        const bodyMat = this._mat(COLORS.body);
        this.bodyMat = bodyMat;
        // Per-arm / per-leg materials (cloned) so a side can be tinted independently.
        this.matArmL = bodyMat.clone();
        this.matArmR = bodyMat.clone();
        this.matLegL = bodyMat.clone();
        this.matLegR = bodyMat.clone();

        const root = new THREE.Group();
        root.position.y = HIP_Y;
        this.scene.add(root);
        this.root = root;

        // Pelvis
        const pelvis = new THREE.Mesh(
            new THREE.CapsuleGeometry(R.pelvis, 0.08, 8, 16),
            bodyMat
        );
        pelvis.rotation.z = Math.PI / 2;
        pelvis.scale.set(1, 1.35, 0.78);
        pelvis.castShadow = true;
        root.add(pelvis);

        // Spine (stoop pivot at pelvis) → torso → chest
        const spineG = group(root, 0, 0, 0);
        this.spineG = spineG;
        const torso = makeBoneUp(spineG, TORSO_LEN, R.torso, bodyMat, { sx: 1.26, sz: 0.74 });
        torso.position.y = TORSO_LEN / 2 + 0.02;

        const chestG = group(spineG, 0, TORSO_LEN, 0);
        this.chestG = chestG;
        makeJointBall(chestG, 0.055, bodyMat, -0.02);

        // Neck (bridges up from the shoulder line) + head
        const neckG = group(chestG, 0, 0.0, 0.006);
        this.neckG = neckG;
        const neck = makeBoneUp(neckG, NECK_LEN, R.neck, bodyMat);
        neck.position.y = NECK_LEN / 2;
        const headG = group(neckG, 0, NECK_LEN, 0.012);
        this.headG = headG;
        const head = new THREE.Mesh(new THREE.SphereGeometry(HEAD_R, 24, 18), bodyMat);
        head.position.y = HEAD_R * 0.86;
        head.scale.set(0.93, 1.12, 1.0);
        head.castShadow = true;
        headG.add(head);

        // Arms
        this.armL = this._buildArm(chestG, -SHOULDER_HALF, this.matArmL, +1);
        this.armR = this._buildArm(chestG, +SHOULDER_HALF, this.matArmR, -1);

        // Legs
        this.legL = this._buildLeg(root, -HIP_HALF, this.matLegL);
        this.legR = this._buildLeg(root, +HIP_HALF, this.matLegR);
    }

    _buildArm(parent, xOffset, mat, side) {
        // side = +1 left, -1 right (used for outward rest angle)
        const shoulder = group(parent, xOffset, -0.02, 0);
        makeJointBall(shoulder, R.upArm * 1.05, mat);
        makeBone(shoulder, UP_ARM, R.upArm, mat);

        const elbow = group(shoulder, 0, -UP_ARM, 0);
        makeJointBall(elbow, R.forearm * 1.05, mat);
        makeBone(elbow, FOREARM, R.forearm, mat);

        const wrist = group(elbow, 0, -FOREARM, 0);
        const hand = makeBone(wrist, HAND_LEN, R.hand, mat, { sx: 1.1, sz: 0.7 });
        hand.position.y = -HAND_LEN / 2;

        // gentle resting abduction so arms hang outward and clear the torso
        shoulder.rotation.z = -side * 0.18;
        elbow.rotation.x = -0.10;   // forearm bends slightly forward at rest (anatomical)
        return { shoulder, elbow, wrist, mat };
    }

    _buildLeg(parent, xOffset, mat) {
        const hip = group(parent, xOffset, -0.04, 0);
        makeJointBall(hip, R.thigh * 0.95, mat);
        makeBone(hip, THIGH, R.thigh, mat);

        const knee = group(hip, 0, -THIGH, 0);
        makeJointBall(knee, R.shin * 1.0, mat);
        makeBone(knee, SHIN, R.shin, mat);

        const ankle = group(knee, 0, -SHIN, 0);
        // foot: a box extending forward (+Z)
        const foot = new THREE.Mesh(
            new THREE.BoxGeometry(0.10, 0.05, FOOT_LEN),
            mat
        );
        foot.position.set(0, -0.025, FOOT_LEN / 2 - 0.05);
        foot.castShadow = true;
        ankle.add(foot);
        return { hip, knee, ankle, mat };
    }

    _initControls() {
        const controls = new OrbitControls(this.camera, this.renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.enablePan = false;
        controls.minDistance = 1.3;
        controls.maxDistance = 8;
        controls.maxPolarAngle = Math.PI * 0.92;
        controls.target.set(0, 0.95, 0);
        controls.addEventListener('start', () => { this._camTweening = false; });
        controls.update();
        this.controls = controls;
    }

    // ── public API ─────────────────────────────────────────────────────────
    /** Apply a pose (all angles radians; missing fields default to neutral). */
    applyPose(p) {
        if (!p) return;
        const r = p.root || {};
        this.root.position.x = r.sway || 0;
        this.root.position.z = r.swayZ || 0;
        this.root.position.y = HIP_Y + (r.bob || 0);
        this.root.rotation.y = r.turn || 0;
        this.root.rotation.x = r.leanAP || 0;
        this.root.rotation.z = r.leanML || 0;

        this.spineG.rotation.x = p.spine || 0;
        this.spineG.rotation.z = p.spineSide || 0;
        this.chestG.rotation.y = p.chestTwist || 0;
        this.neckG.rotation.x = p.neck || 0;
        this.headG.rotation.x = p.head || 0;

        this._arm(this.armL, p.arms && p.arms.l, +1);
        this._arm(this.armR, p.arms && p.arms.r, -1);
        this._leg(this.legL, p.legs && p.legs.l);
        this._leg(this.legR, p.legs && p.legs.r);
    }

    _arm(arm, a, side) {
        a = a || {};
        arm.shoulder.rotation.x = a.shoulder || 0;
        // Negative-side keeps both arms abducted *outward*; shoulderOut adds more.
        arm.shoulder.rotation.z = -side * (0.18 + (a.shoulderOut || 0));
        // Elbow flexion bends the forearm FORWARD (+Z); rotation about +X would hyperextend it.
        arm.elbow.rotation.x = -(0.10 + (a.elbow || 0));
        arm.wrist.rotation.x = a.tremor || 0;
        arm.wrist.rotation.z = a.tremorZ || 0;
    }

    _leg(leg, l) {
        l = l || {};
        leg.hip.rotation.x = l.hip || 0;
        leg.hip.rotation.z = l.hipOut || 0;
        leg.knee.rotation.x = l.knee || 0;
        leg.ankle.rotation.x = l.ankle || 0;
    }

    /** Tint a side toward the accent colour to mark the more-affected limb. */
    setAffected(side /* 'l'|'r'|null */, amount = 0) {
        const accent = new THREE.Color(COLORS.accent);
        const base = new THREE.Color(COLORS.body);
        const set = (mat, on) => mat.color.copy(base).lerp(accent, on ? Math.min(amount, 1) : 0);
        set(this.matArmL, side === 'l');
        set(this.matArmR, side === 'r');
        set(this.matLegL, side === 'l');
        set(this.matLegR, side === 'r');
    }

    setView(name, immediate = false) {
        const views = {
            threeQuarter: [2.4, 1.7, 3.0],
            front:        [0, 1.15, 3.5],
            side:         [3.5, 1.05, 0.001],
            top:          [0.001, 4.4, 0.001],
        };
        if (name === 'reset') name = 'threeQuarter';
        const v = views[name] || views.threeQuarter;
        this._camDest.set(v[0], v[1], v[2]);
        this.controls.target.set(0, 0.95, 0);
        if (immediate) {
            this.camera.position.copy(this._camDest);
            this.controls.update();
            this._camTweening = false;
        } else {
            this._camTweening = true;
        }
    }

    start() {
        if (this._raf) return;
        const loop = (t) => {
            this._raf = requestAnimationFrame(loop);
            const dt = this._lastT ? (t - this._lastT) / 1000 : 0;
            this._lastT = t;
            if (this.onFrame) this.onFrame(Math.min(dt, 0.05));
            if (this._camTweening) {
                this.camera.position.lerp(this._camDest, 0.10);
                if (this.camera.position.distanceTo(this._camDest) < 0.01) this._camTweening = false;
            }
            this.controls.update();
            this.renderer.render(this.scene, this.camera);
        };
        this._raf = requestAnimationFrame(loop);
    }

    _resize() {
        const w = this.container.clientWidth, h = this.container.clientHeight;
        if (!w || !h) return;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }

    dispose() {
        if (this._raf) cancelAnimationFrame(this._raf);
        this._ro.disconnect();
        window.removeEventListener('resize', this._onResize);
        this.renderer.dispose();
        if (this.renderer.domElement.parentNode) {
            this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
        }
    }
}

// torso bone grows upward from the spine pivot (mirror of makeBone)
function makeBoneUp(parent, length, radius, material, { sx = 1, sz = 1 } = {}) {
    const cyl = Math.max(length - 1.4 * radius, 0.01);
    const mesh = new THREE.Mesh(new THREE.CapsuleGeometry(radius, cyl, 8, 18), material);
    mesh.scale.set(sx, 1, sz);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    parent.add(mesh);
    return mesh;
}
