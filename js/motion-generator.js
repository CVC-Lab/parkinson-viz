/**
 * Motion Silhouette Generator - JavaScript Port
 * Generates anatomically-correct motion silhouettes based on patient data
 */

export class MotionSilhouetteGenerator {
    constructor() {
        this.silhouetteBase = this.createAnatomicalSilhouette();
        this.bodyColors = {
            'head': '#3498db', 'neck': '#2980b9', 'torso': '#2c3e50',
            'left_upper_arm': '#e74c3c', 'left_forearm': '#c0392b', 'left_hand': '#a93226',
            'right_upper_arm': '#27ae60', 'right_forearm': '#229954', 'right_hand': '#1e8449',
            'left_thigh': '#f39c12', 'left_shin': '#e67e22', 'left_foot': '#d35400',
            'right_thigh': '#9b59b6', 'right_shin': '#8e44ad', 'right_foot': '#7d3c98'
        };
    }

    createAnatomicalSilhouette() {
        // Using 8-head figure proportions with realistic body shape
        return {
            head: {
                x: [-0.5, -0.4, -0.2, 0, 0.2, 0.4, 0.5, 0.4, 0.2, 0, -0.2, -0.4, -0.5],
                y: [7.2, 7.5, 7.8, 8.0, 7.8, 7.5, 7.2, 6.8, 6.5, 6.3, 6.5, 6.8, 7.2]
            },
            neck: {
                x: [-0.2, 0.2, 0.2, -0.2, -0.2],
                y: [6.3, 6.3, 6.0, 6.0, 6.3]
            },
            torso: {
                x: [-1.0, -1.2, -1.1, -0.9, -0.7, 0.7, 0.9, 1.1, 1.2, 1.0, 0.8, -0.8, -1.0],
                y: [6.0, 5.5, 4.0, 2.5, 2.0, 2.0, 2.5, 4.0, 5.5, 6.0, 6.0, 6.0, 6.0]
            },
            left_upper_arm: {
                x: [-1.0, -1.8, -2.0, -1.5, -1.0],
                y: [5.5, 4.5, 3.8, 3.5, 4.2]
            },
            left_forearm: {
                x: [-1.5, -2.2, -2.5, -2.0, -1.5],
                y: [3.5, 2.8, 2.0, 1.8, 2.5]
            },
            left_hand: {
                x: [-2.0, -2.3, -2.4, -2.1, -2.0],
                y: [1.8, 1.5, 1.2, 1.0, 1.3]
            },
            right_upper_arm: {
                x: [1.0, 1.8, 2.0, 1.5, 1.0],
                y: [5.5, 4.5, 3.8, 3.5, 4.2]
            },
            right_forearm: {
                x: [1.5, 2.2, 2.5, 2.0, 1.5],
                y: [3.5, 2.8, 2.0, 1.8, 2.5]
            },
            right_hand: {
                x: [2.0, 2.3, 2.4, 2.1, 2.0],
                y: [1.8, 1.5, 1.2, 1.0, 1.3]
            },
            left_thigh: {
                x: [-0.7, -0.9, -1.0, -0.8, -0.5, -0.7],
                y: [2.0, 1.5, 0.5, -0.5, 0.2, 2.0]
            },
            left_shin: {
                x: [-0.8, -1.0, -1.1, -0.9, -0.7, -0.8],
                y: [-0.5, -1.5, -2.8, -3.0, -1.8, -0.5]
            },
            left_foot: {
                x: [-0.9, -1.2, -0.6, -0.5, -0.7, -0.9],
                y: [-3.0, -3.2, -3.3, -3.0, -2.8, -3.0]
            },
            right_thigh: {
                x: [0.7, 0.9, 1.0, 0.8, 0.5, 0.7],
                y: [2.0, 1.5, 0.5, -0.5, 0.2, 2.0]
            },
            right_shin: {
                x: [0.8, 1.0, 1.1, 0.9, 0.7, 0.8],
                y: [-0.5, -1.5, -2.8, -3.0, -1.8, -0.5]
            },
            right_foot: {
                x: [0.9, 1.2, 0.6, 0.5, 0.7, 0.9],
                y: [-3.0, -3.2, -3.3, -3.0, -2.8, -3.0]
            }
        };
    }

    generateMotionFrame(patientData, motionType = 'gait', timePhase = 0) {
        try {
            const leftArmAmp = this.getValue(patientData, 'LA_AMP_U', 30);
            const rightArmAmp = this.getValue(patientData, 'RA_AMP_U', 30);
            const gaitSpeed = this.getValue(patientData, 'SP_U', 1.0);
            const asymmetry = this.getValue(patientData, 'ASA_U', 0.1);

            let modifications;
            if (motionType === 'gait') {
                modifications = this.calculateGaitMotion(leftArmAmp, rightArmAmp, gaitSpeed, timePhase, asymmetry);
            } else if (motionType === 'tug') {
                const tugPhase = this.determineTugPhase(patientData, timePhase);
                modifications = this.calculateTugMotion(leftArmAmp, rightArmAmp, gaitSpeed, tugPhase, timePhase);
            } else if (motionType === 'balance') {
                const sensorMean = this.getValue(patientData, 'SENSOR_MEAN', 0);
                modifications = this.calculateBalanceMotion(sensorMean, timePhase);
            } else {
                modifications = this.calculateGaitMotion(leftArmAmp, rightArmAmp, gaitSpeed, timePhase, asymmetry);
            }

            return this.applyMotionModifications(modifications);
        } catch (error) {
            console.error('Error in generateMotionFrame:', error);
            return JSON.parse(JSON.stringify(this.silhouetteBase)); // Deep copy
        }
    }

    getValue(data, key, defaultValue) {
        if (!data || data[key] === undefined || data[key] === null || isNaN(data[key])) {
            return defaultValue;
        }
        return parseFloat(data[key]);
    }

    calculateGaitMotion(leftArmAmp, rightArmAmp, gaitSpeed, timePhase, asymmetry) {
        const leftSwing = (leftArmAmp / 50.0) * Math.sin(timePhase) * 0.5;
        const rightSwing = (rightArmAmp / 50.0) * Math.sin(timePhase + Math.PI) * 0.5;

        const asymmetryFactor = Math.min(asymmetry / 10.0, 0.3);
        const leftSwingAdj = leftSwing * (1 + asymmetryFactor);
        const rightSwingAdj = rightSwing * (1 - asymmetryFactor);

        const speedFactor = Math.min(gaitSpeed, 1.5);
        const legPhase = timePhase * speedFactor;

        const leftLegSwing = Math.sin(legPhase + Math.PI) * 0.3;
        const rightLegSwing = Math.sin(legPhase) * 0.3;

        return {
            left_arm_swing: leftSwingAdj,
            right_arm_swing: rightSwingAdj,
            left_leg_swing: leftLegSwing,
            right_leg_swing: rightLegSwing,
            torso_lean: asymmetryFactor * 0.02,
            head_bob: Math.sin(timePhase * 2) * 0.01,
            torso_rotation: (leftSwingAdj - rightSwingAdj) * 0.05
        };
    }

    calculateBalanceMotion(sensorMean, timePhase) {
        const swayMagnitude = Math.min(Math.abs(sensorMean) / 50.0, 0.3);
        const anteriorPosteriorSway = swayMagnitude * Math.sin(timePhase * 0.8);
        const medialLateralSway = swayMagnitude * Math.cos(timePhase * 0.6);

        return {
            torso_sway_ap: anteriorPosteriorSway,
            torso_sway_ml: medialLateralSway,
            left_arm_swing: medialLateralSway * 0.3,
            right_arm_swing: -medialLateralSway * 0.3,
            left_leg_swing: 0,
            right_leg_swing: 0,
            head_bob: anteriorPosteriorSway * 0.5
        };
    }

    calculateTugMotion(leftArmAmp, rightArmAmp, gaitSpeed, tugPhase, timePhase) {
        if (tugPhase === 'sitting') {
            return this.getSittingPosture();
        } else if (tugPhase === 'standing') {
            return this.getStandingTransition(timePhase);
        } else if (tugPhase === 'turning') {
            return this.getTurningMotion(timePhase, gaitSpeed);
        } else {
            return this.calculateGaitMotion(leftArmAmp, rightArmAmp, gaitSpeed, timePhase, 0.1);
        }
    }

    determineTugPhase(patientData, timePhase) {
        const normalizedTime = (timePhase % (2 * Math.PI)) / (2 * Math.PI);

        if (normalizedTime < 0.1) return 'sitting';
        if (normalizedTime < 0.2) return 'standing';
        if (normalizedTime < 0.4) return 'walking_straight';
        if (normalizedTime < 0.6) return 'turning';
        if (normalizedTime < 0.9) return 'walking_straight';
        return 'sitting';
    }

    getSittingPosture() {
        return {
            torso_lean: 0.1,
            left_arm_swing: 0,
            right_arm_swing: 0,
            left_leg_swing: 0,
            right_leg_swing: 0,
            head_bob: 0
        };
    }

    getStandingTransition(timePhase) {
        const transitionFactor = Math.sin(timePhase * 3) * 0.2;
        return {
            torso_lean: -transitionFactor,
            left_arm_swing: transitionFactor * 0.3,
            right_arm_swing: transitionFactor * 0.3,
            left_leg_swing: 0,
            right_leg_swing: 0,
            head_bob: transitionFactor * 0.5
        };
    }

    getTurningMotion(timePhase, gaitSpeed) {
        const turnFactor = Math.sin(timePhase * 2) * gaitSpeed * 0.3;
        return {
            torso_rotation: turnFactor * 0.5,
            left_arm_swing: turnFactor,
            right_arm_swing: -turnFactor,
            left_leg_swing: turnFactor * 0.4,
            right_leg_swing: -turnFactor * 0.4,
            head_bob: Math.abs(turnFactor) * 0.2
        };
    }

    applyMotionModifications(mods) {
        const silhouette = {};
        const base = JSON.parse(JSON.stringify(this.silhouetteBase)); // Deep copy

        const leftArmSwing = mods.left_arm_swing || 0;
        const rightArmSwing = mods.right_arm_swing || 0;
        const leftLegSwing = mods.left_leg_swing || 0;
        const rightLegSwing = mods.right_leg_swing || 0;
        const torsoLean = mods.torso_lean || 0;
        const torsoSwayAp = mods.torso_sway_ap || 0;
        const torsoSwayMl = mods.torso_sway_ml || 0;
        const headBob = mods.head_bob || 0;
        const torsoRotation = mods.torso_rotation || 0;

        // Head
        silhouette.head = {
            x: base.head.x.map(x => x + torsoSwayMl * 0.3),
            y: base.head.y.map(y => y + torsoLean * 0.2 + headBob)
        };

        // Neck
        silhouette.neck = {
            x: base.neck.x.map(x => x + torsoSwayMl * 0.4),
            y: base.neck.y.map(y => y + torsoLean * 0.3)
        };

        // Torso
        silhouette.torso = {
            x: base.torso.x.map(x => x + torsoSwayMl * 0.5),
            y: base.torso.y.map(y => y + torsoLean * 0.5)
        };

        // Left arm
        silhouette.left_upper_arm = {
            x: base.left_upper_arm.x.map(x => x - leftArmSwing * 0.5),
            y: base.left_upper_arm.y.map(y => y + leftArmSwing * 0.2)
        };
        silhouette.left_forearm = {
            x: base.left_forearm.x.map(x => x - leftArmSwing * 0.8),
            y: base.left_forearm.y.map(y => y + leftArmSwing * 0.3)
        };
        silhouette.left_hand = {
            x: base.left_hand.x.map(x => x - leftArmSwing * 1.0),
            y: base.left_hand.y.map(y => y + leftArmSwing * 0.4)
        };

        // Right arm
        silhouette.right_upper_arm = {
            x: base.right_upper_arm.x.map(x => x - rightArmSwing * 0.5),
            y: base.right_upper_arm.y.map(y => y + rightArmSwing * 0.2)
        };
        silhouette.right_forearm = {
            x: base.right_forearm.x.map(x => x - rightArmSwing * 0.8),
            y: base.right_forearm.y.map(y => y + rightArmSwing * 0.3)
        };
        silhouette.right_hand = {
            x: base.right_hand.x.map(x => x - rightArmSwing * 1.0),
            y: base.right_hand.y.map(y => y + rightArmSwing * 0.4)
        };

        // Legs
        const legParts = [
            ['left_thigh', 'left_shin', 'left_foot', leftLegSwing],
            ['right_thigh', 'right_shin', 'right_foot', rightLegSwing]
        ];

        for (const [thigh, shin, foot, swing] of legParts) {
            for (const part of [thigh, shin, foot]) {
                silhouette[part] = {
                    x: base[part].x.map(x => x + swing * 0.3),
                    y: base[part].y
                };
            }
        }

        return silhouette;
    }

    getBodyColors() {
        return this.bodyColors;
    }
}
