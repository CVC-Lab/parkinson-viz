# app.py - FIXED MULTI-DATASET ENHANCED PARKINSON'S VISUALIZATION WITH WORKING SILHOUETTE MOTION

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from scipy import stats
import math
import copy
import warnings
warnings.filterwarnings('ignore')

class ParkinsonDataLoader:
    """Enhanced data loader for multi-modal Parkinson's datasets"""
    def __init__(self, base_path='.'):
        self.base_path = base_path
        self.data = {}
        
    def load_all_datasets(self):
        """Load and merge all relevant datasets"""
        print("Loading datasets...")
        
        # Load motor assessments - Added comprehensive motor assessment loading
        self.load_gait_data()
        self.load_updrs_data()
        self.load_demographics()
        self.load_digital_sensor_data()
        
        # Merge datasets - Sophisticated multi-dataset integration
        self.merge_datasets()
        
        return self.data['merged']
    
    def load_gait_data(self):
        """Load gait and arm swing data"""
        try:
            gait_path = os.path.join(self.base_path, 'Motor_Assessments', 'Gait_Data___Arm_swing_06Jan2025.csv')
            self.data['gait'] = pd.read_csv(gait_path)
            print(f"✓ Loaded gait data: {self.data['gait'].shape}")
        except Exception as e:
            print(f"❌ Error loading gait data: {e}")
            self.data['gait'] = pd.DataFrame()
    
    def load_updrs_data(self):
        """Load UPDRS clinical scores - Added all UPDRS parts for comprehensive clinical assessment"""
        try:
            # UPDRS Part III (Motor examination)
            updrs3_path = os.path.join(self.base_path, 'Motor_Assessments', 'MDS-UPDRS_Part_III_06Jan2025.csv')
            updrs3 = pd.read_csv(updrs3_path)
            
            # Keep only essential columns to avoid memory issues - Selected key clinical indicators
            updrs3_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'NP3TOT', 'NP3SPCH', 'NP3FACXP', 'NP3RIGN', 
                          'NP3RIGRU', 'NP3RIGLU', 'NP3FTAPR', 'NP3FTAPL', 'NP3GAIT', 'NP3PSTBL', 
                          'NP3BRADY', 'NP3PTRMR', 'NP3PTRML', 'NHY']
            self.data['updrs3'] = updrs3[updrs3_cols].copy()
            
            # UPDRS Part II (Patient questionnaire) - Added patient-reported outcomes
            updrs2_path = os.path.join(self.base_path, 'Motor_Assessments', 'MDS_UPDRS_Part_II__Patient_Questionnaire_06Jan2025.csv')
            updrs2 = pd.read_csv(updrs2_path)
            
            updrs2_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'NP2PTOT', 'NP2SPCH', 'NP2WALK', 'NP2TURN', 'NP2TRMR']
            self.data['updrs2'] = updrs2[updrs2_cols].copy()
            
            print(f"✓ Loaded UPDRS III: {self.data['updrs3'].shape}")
            print(f"✓ Loaded UPDRS II: {self.data['updrs2'].shape}")
            
        except Exception as e:
            print(f"❌ Error loading UPDRS data: {e}")
            self.data['updrs3'] = pd.DataFrame()
            self.data['updrs2'] = pd.DataFrame()
    
    def load_demographics(self):
        """Load demographics and participant status - Added demographic stratification variables"""
        try:
            # Demographics
            demo_path = os.path.join(self.base_path, 'Subject_Characteristics', 'Demographics_08Jan2025.csv')
            demographics = pd.read_csv(demo_path)
            demo_cols = ['PATNO', 'SEX', 'BIRTHDT', 'HANDED', 'HISPLAT', 'RAWHITE', 'RABLACK', 'RAASIAN']
            self.data['demographics'] = demographics[demo_cols].copy()
            
            # Participant status - Added cohort and enrollment information
            status_path = os.path.join(self.base_path, 'Subject_Characteristics', 'Participant_Status_08Jan2025.csv')
            status = pd.read_csv(status_path)
            status_cols = ['PATNO', 'COHORT', 'COHORT_DEFINITION', 'ENROLL_AGE', 'ENROLL_STATUS']
            self.data['status'] = status[status_cols].copy()
            
            print(f"✓ Loaded demographics: {self.data['demographics'].shape}")
            print(f"✓ Loaded status: {self.data['status'].shape}")
            
        except Exception as e:
            print(f"❌ Error loading demographics: {e}")
            self.data['demographics'] = pd.DataFrame()
            self.data['status'] = pd.DataFrame()
    
    def load_digital_sensor_data(self):
        """Load digital sensor summary data - Added high-frequency behavioral data"""
        try:
            sensor_path = os.path.join(self.base_path, 'Digital_Sensor', 'Roche_PD_Monitoring_App_v2_data_06Jan2025.csv')
            sensor_data = pd.read_csv(sensor_path)
            
            # Focus on key sensor metrics - Selected clinically relevant sensor features
            sensor_cols = ['PATNO', 'QRSSCAT', 'QRSTEST', 'QRSRESN', 'Age']
            self.data['sensors'] = sensor_data[sensor_cols].copy()
            
            # Aggregate sensor data by patient - Created sensor summary metrics
            sensor_summary = self.data['sensors'].groupby('PATNO').agg({
                'QRSRESN': ['mean', 'std', 'count'],
                'Age': 'first'
            }).reset_index()
            
            # Flatten column names
            sensor_summary.columns = ['PATNO', 'SENSOR_MEAN', 'SENSOR_STD', 'SENSOR_COUNT', 'SENSOR_AGE']
            self.data['sensor_summary'] = sensor_summary
            
            print(f"✓ Loaded sensor data: {self.data['sensors'].shape}")
            print(f"✓ Created sensor summary: {self.data['sensor_summary'].shape}")
            
        except Exception as e:
            print(f"❌ Error loading sensor data: {e}")
            self.data['sensors'] = pd.DataFrame()
            self.data['sensor_summary'] = pd.DataFrame()
    
    def merge_datasets(self):
        """Merge all datasets on PATNO and EVENT_ID - Comprehensive multi-modal data integration"""
        if self.data['gait'].empty:
            print("❌ No gait data available for merging")
            self.data['merged'] = pd.DataFrame()
            return
        
        # Start with gait data as base - Using objective motor measurements as foundation
        merged = self.data['gait'].copy()
        
        # Add UPDRS scores - Integrated clinical severity assessments
        if not self.data['updrs3'].empty:
            merged = merged.merge(self.data['updrs3'], on=['PATNO', 'EVENT_ID'], how='left', suffixes=('', '_updrs3'))
        
        if not self.data['updrs2'].empty:
            merged = merged.merge(self.data['updrs2'], on=['PATNO', 'EVENT_ID'], how='left', suffixes=('', '_updrs2'))
        
        # Add demographics (patient-level) - Added demographic stratification
        if not self.data['demographics'].empty:
            merged = merged.merge(self.data['demographics'], on='PATNO', how='left')
        
        if not self.data['status'].empty:
            merged = merged.merge(self.data['status'], on='PATNO', how='left')
        
        # Add sensor summary (patient-level) - Integrated digital biomarkers
        if not self.data['sensor_summary'].empty:
            merged = merged.merge(self.data['sensor_summary'], on='PATNO', how='left')
        
        # Clean and enhance merged dataset - Comprehensive data quality improvements
        self.enhance_merged_data(merged)
        
        print(f"✓ Final merged dataset: {self.data['merged'].shape}")
        print(f"✓ Available patients: {self.data['merged']['PATNO'].nunique()}")
        print(f"✓ Available cohorts: {self.data['merged']['COHORT_DEFINITION'].value_counts().to_dict()}")
    
    def enhance_merged_data(self, merged):
        """Enhance merged dataset with derived features - Added sophisticated clinical feature engineering"""
        
        # Basic cleaning
        merged = merged.dropna(subset=['PATNO', 'ASA_U', 'SP_U'])
        
        # Enhanced cohort mapping - More comprehensive cohort classification
        if 'COHORT_DEFINITION' in merged.columns:
            merged['COHORT_NAME'] = merged['COHORT_DEFINITION'].fillna('Unknown')
        else:
            # Fallback to numeric cohort mapping
            cohort_map = {1: 'PD', 2: 'Healthy Control', 3: 'Prodromal', 4: 'SWEDD'}
            merged['COHORT_NAME'] = merged['COHORT'].map(cohort_map).fillna('Unknown')
        
        # Enhanced motor features - Added comprehensive movement analysis
        merged['ARM_ASYMMETRY'] = abs(merged['RA_AMP_U'] - merged['LA_AMP_U']) / (merged['RA_AMP_U'] + merged['LA_AMP_U'] + 1e-6)
        merged['TOTAL_JERK'] = merged['R_JERK_U'] + merged['L_JERK_U']
        merged['JERK_ASYMMETRY'] = abs(merged['R_JERK_U'] - merged['L_JERK_U']) / (merged['R_JERK_U'] + merged['L_JERK_U'] + 1e-6)
        
        # Clinical severity composites - Multi-domain severity assessment
        if 'NP3TOT' in merged.columns:
            merged['CLINICAL_MOTOR_SEVERITY'] = merged['NP3TOT'].fillna(0)
        else:
            merged['CLINICAL_MOTOR_SEVERITY'] = 0
        
        if 'NP2PTOT' in merged.columns:
            merged['PATIENT_REPORTED_SEVERITY'] = merged['NP2PTOT'].fillna(0)
        else:
            merged['PATIENT_REPORTED_SEVERITY'] = 0
        
        # Multi-modal composite scores - Integration of sensor, clinical, and objective measures
        merged['OBJECTIVE_MOTOR_SCORE'] = (
            (merged['ASA_U'] / (merged['ASA_U'].std() + 1e-6)) +
            (1 / (merged['SP_U'] + 0.1)) +
            (merged['TOTAL_JERK'] / (merged['TOTAL_JERK'].std() + 1e-6))
        )
        
        # Sensor-clinical correlation - Multi-modal biomarker integration
        if 'SENSOR_MEAN' in merged.columns:
            merged['SENSOR_CLINICAL_RATIO'] = merged['SENSOR_MEAN'] / (merged['CLINICAL_MOTOR_SEVERITY'] + 1)
        else:
            merged['SENSOR_CLINICAL_RATIO'] = np.nan
        
        # Age and sex adjustments - Demographic normalization
        if 'ENROLL_AGE' in merged.columns:
            merged['AGE_ADJUSTED_SEVERITY'] = merged['CLINICAL_MOTOR_SEVERITY'] / (merged['ENROLL_AGE'] / 65.0)
        else:
            merged['AGE_ADJUSTED_SEVERITY'] = merged['CLINICAL_MOTOR_SEVERITY']
        
        # Movement quality indices - Comprehensive movement characterization
        merged['MOVEMENT_QUALITY'] = merged['SP_U'] / (merged['TOTAL_JERK'] + 1)
        merged['BILATERAL_COORDINATION'] = 1 - merged['ARM_ASYMMETRY']  # Higher = better coordination
        
        # Clinical categories - Evidence-based clinical staging
        merged['SEVERITY_CATEGORY'] = pd.cut(
            merged['CLINICAL_MOTOR_SEVERITY'], 
            bins=[-1, 0, 20, 40, 100], 
            labels=['No Data', 'Mild', 'Moderate', 'Severe']
        )
        
        merged['SPEED_CATEGORY'] = pd.cut(
            merged['SP_U'], 
            bins=[0, 0.8, 1.0, 1.2, float('inf')], 
            labels=['Very Slow', 'Slow', 'Normal', 'Fast']
        )
        
        self.data['merged'] = merged

# FIXED: Enhanced Motion Silhouette Visualization System - Anatomically-accurate with proper animation
class MotionSilhouetteGenerator:
    """Generate anatomically-correct motion silhouettes based on patient data with proper animation"""
    
    def __init__(self):
        # Define anatomically-accurate silhouette coordinates - Based on your reference image
        self.silhouette_base = self.create_anatomical_silhouette()
        
    def create_anatomical_silhouette(self):
        """Create anatomical human silhouette with proper proportions like your reference image"""
        # Using 8-head figure proportions with more realistic body shape
        return {
            'head': {
                'x': [-0.5, -0.4, -0.2, 0, 0.2, 0.4, 0.5, 0.4, 0.2, 0, -0.2, -0.4, -0.5], 
                'y': [7.2, 7.5, 7.8, 8.0, 7.8, 7.5, 7.2, 6.8, 6.5, 6.3, 6.5, 6.8, 7.2]
            },
            'neck': {
                'x': [-0.2, 0.2, 0.2, -0.2, -0.2],
                'y': [6.3, 6.3, 6.0, 6.0, 6.3]
            },
            'torso': {
                'x': [-1.0, -1.2, -1.1, -0.9, -0.7, 0.7, 0.9, 1.1, 1.2, 1.0, 0.8, -0.8, -1.0], 
                'y': [6.0, 5.5, 4.0, 2.5, 2.0, 2.0, 2.5, 4.0, 5.5, 6.0, 6.0, 6.0, 6.0]
            },
            'left_upper_arm': {
                'x': [-1.0, -1.8, -2.0, -1.5, -1.0],
                'y': [5.5, 4.5, 3.8, 3.5, 4.2]
            },
            'left_forearm': {
                'x': [-1.5, -2.2, -2.5, -2.0, -1.5],
                'y': [3.5, 2.8, 2.0, 1.8, 2.5]
            },
            'left_hand': {
                'x': [-2.0, -2.3, -2.4, -2.1, -2.0],
                'y': [1.8, 1.5, 1.2, 1.0, 1.3]
            },
            'right_upper_arm': {
                'x': [1.0, 1.8, 2.0, 1.5, 1.0],
                'y': [5.5, 4.5, 3.8, 3.5, 4.2]
            },
            'right_forearm': {
                'x': [1.5, 2.2, 2.5, 2.0, 1.5],
                'y': [3.5, 2.8, 2.0, 1.8, 2.5]
            },
            'right_hand': {
                'x': [2.0, 2.3, 2.4, 2.1, 2.0],
                'y': [1.8, 1.5, 1.2, 1.0, 1.3]
            },
            'left_thigh': {
                'x': [-0.7, -0.9, -1.0, -0.8, -0.5, -0.7],
                'y': [2.0, 1.5, 0.5, -0.5, 0.2, 2.0]
            },
            'left_shin': {
                'x': [-0.8, -1.0, -1.1, -0.9, -0.7, -0.8],
                'y': [-0.5, -1.5, -2.8, -3.0, -1.8, -0.5]
            },
            'left_foot': {
                'x': [-0.9, -1.2, -0.6, -0.5, -0.7, -0.9],
                'y': [-3.0, -3.2, -3.3, -3.0, -2.8, -3.0]
            },
            'right_thigh': {
                'x': [0.7, 0.9, 1.0, 0.8, 0.5, 0.7],
                'y': [2.0, 1.5, 0.5, -0.5, 0.2, 2.0]
            },
            'right_shin': {
                'x': [0.8, 1.0, 1.1, 0.9, 0.7, 0.8],
                'y': [-0.5, -1.5, -2.8, -3.0, -1.8, -0.5]
            },
            'right_foot': {
                'x': [0.9, 1.2, 0.6, 0.5, 0.7, 0.9],
                'y': [-3.0, -3.2, -3.3, -3.0, -2.8, -3.0]
            }
        }
    
    def generate_motion_frame(self, patient_data, motion_type='gait', time_phase=0):
        """Generate a single frame with better error handling"""
        
        try:
            # Extract motion parameters with defaults and error checking
            left_arm_amp = patient_data.get('LA_AMP_U', 30) if patient_data else 30
            right_arm_amp = patient_data.get('RA_AMP_U', 30) if patient_data else 30
            gait_speed = patient_data.get('SP_U', 1.0) if patient_data else 1.0
            asymmetry = patient_data.get('ASA_U', 0.1) if patient_data else 0.1
            
            # Ensure numeric values and handle NaN
            left_arm_amp = float(left_arm_amp) if not pd.isna(left_arm_amp) else 30
            right_arm_amp = float(right_arm_amp) if not pd.isna(right_arm_amp) else 30
            gait_speed = float(gait_speed) if not pd.isna(gait_speed) else 1.0
            asymmetry = float(asymmetry) if not pd.isna(asymmetry) else 0.1
            
            # Calculate motion modifications
            if motion_type == 'gait':
                modifications = self.calculate_gait_motion(
                    left_arm_amp, right_arm_amp, gait_speed, time_phase, asymmetry
                )
            elif motion_type == 'tug':
                tug_phase = self.determine_tug_phase(patient_data, time_phase)
                modifications = self.calculate_tug_motion(
                    left_arm_amp, right_arm_amp, gait_speed, tug_phase, time_phase
                )
            elif motion_type == 'balance':
                sensor_mean = patient_data.get('SENSOR_MEAN', 0) if patient_data else 0
                modifications = self.calculate_balance_motion(sensor_mean, time_phase)
            else:
                modifications = self.calculate_gait_motion(
                    left_arm_amp, right_arm_amp, gait_speed, time_phase, asymmetry
                )
            
            # Apply modifications to base silhouette
            animated_silhouette = self.apply_motion_modifications(modifications)
            
            return animated_silhouette
            
        except Exception as e:
            print(f"Error in generate_motion_frame: {e}")
            # Return static silhouette on error
            return self.silhouette_base.copy()
    
    def calculate_gait_motion(self, left_arm_amp, right_arm_amp, gait_speed, time_phase, asymmetry):
        # Normalize amplitudes to realistic range (0.1 to 0.4 radians)
        left_swing = (left_arm_amp / 50.0) * math.sin(time_phase) * 0.5  # Increased from 0.15
        right_swing = (right_arm_amp / 50.0) * math.sin(time_phase + math.pi) * 0.5  # Increased from 0.15
        
        # Apply asymmetry more subtly
        asymmetry_factor = min(asymmetry / 10.0, 0.3)  # Reduce asymmetry effect
        left_swing *= (1 + asymmetry_factor)
        right_swing *= (1 - asymmetry_factor)
        
        # More natural leg motion
        speed_factor = min(gait_speed, 1.5)
        leg_phase = time_phase * speed_factor
        
        left_leg_swing = math.sin(leg_phase + math.pi) * 0.3  # Much more visible movement
        right_leg_swing = math.sin(leg_phase) * 0.3
        
        return {
            'left_arm_swing': left_swing,
            'right_arm_swing': right_swing,
            'left_leg_swing': left_leg_swing,
            'right_leg_swing': right_leg_swing,
            'torso_lean': asymmetry_factor * 0.02,  # Very subtle torso lean
            'head_bob': math.sin(time_phase * 2) * 0.01,  # Minimal head bob
            'torso_rotation': (left_swing - right_swing) * 0.05  # Subtle rotation
        }
    
    def calculate_balance_motion(self, sensor_mean, time_phase):
        """Calculate postural sway motion - FIXED balance animation"""
        
        # Use sensor data to determine sway magnitude
        sway_magnitude = abs(sensor_mean) / 50.0 if sensor_mean else 0.1
        sway_magnitude = min(sway_magnitude, 0.3)  # Cap maximum sway
        
        # Multi-directional sway pattern
        anterior_posterior_sway = sway_magnitude * math.sin(time_phase * 0.8)
        medial_lateral_sway = sway_magnitude * math.cos(time_phase * 0.6)
        
        return {
            'torso_sway_ap': anterior_posterior_sway,
            'torso_sway_ml': medial_lateral_sway,
            'left_arm_swing': medial_lateral_sway * 0.3,  # Compensatory arm movement
            'right_arm_swing': -medial_lateral_sway * 0.3,
            'left_leg_swing': 0,
            'right_leg_swing': 0,
            'head_bob': anterior_posterior_sway * 0.5
        }
    
    def calculate_tug_motion(self, left_arm_amp, right_arm_amp, gait_speed, tug_phase, time_phase):
        """Calculate TUG test motion - FIXED TUG animation"""
        
        if tug_phase == 'sitting':
            return self.get_sitting_posture()
        elif tug_phase == 'standing':
            return self.get_standing_transition(time_phase)
        elif tug_phase == 'turning':
            return self.get_turning_motion(time_phase, gait_speed)
        else:  # walking phases
            return self.calculate_gait_motion(left_arm_amp, right_arm_amp, gait_speed, time_phase, 0.1)
    
    def determine_tug_phase(self, patient_data, time_phase):
        """Determine TUG test phase - FIXED phase determination"""
        
        # Normalize time_phase to 0-1 cycle
        normalized_time = (time_phase % (2 * math.pi)) / (2 * math.pi)
        
        if normalized_time < 0.1:
            return 'sitting'
        elif normalized_time < 0.2:
            return 'standing'
        elif normalized_time < 0.4:
            return 'walking_straight'
        elif normalized_time < 0.6:
            return 'turning'
        elif normalized_time < 0.9:
            return 'walking_straight'
        else:
            return 'sitting'
    
    def get_sitting_posture(self):
        """Return sitting posture modifications"""
        return {
            'torso_lean': 0.1,
            'left_arm_swing': 0,
            'right_arm_swing': 0,
            'left_leg_swing': 0,
            'right_leg_swing': 0,
            'head_bob': 0
        }
    
    def get_standing_transition(self, time_phase):
        """Return standing transition motion"""
        transition_factor = math.sin(time_phase * 3) * 0.2
        return {
            'torso_lean': -transition_factor,
            'left_arm_swing': transition_factor * 0.3,
            'right_arm_swing': transition_factor * 0.3,
            'left_leg_swing': 0,
            'right_leg_swing': 0,
            'head_bob': transition_factor * 0.5
        }
    
    def get_turning_motion(self, time_phase, gait_speed):
        """Return turning motion pattern"""
        turn_factor = math.sin(time_phase * 2) * gait_speed * 0.3
        return {
            'torso_rotation': turn_factor * 0.5,
            'left_arm_swing': turn_factor,
            'right_arm_swing': -turn_factor,
            'left_leg_swing': turn_factor * 0.4,
            'right_leg_swing': -turn_factor * 0.4,
            'head_bob': abs(turn_factor) * 0.2
        }
    
    def apply_motion_modifications(self, modifications):
        """Apply calculated motion modifications to base silhouette with natural constraints"""
        
        silhouette = {}
        base = copy.deepcopy(self.silhouette_base)
        
        # Get modification values with defaults
        left_arm_swing = modifications.get('left_arm_swing', 0)
        right_arm_swing = modifications.get('right_arm_swing', 0)
        left_leg_swing = modifications.get('left_leg_swing', 0)
        right_leg_swing = modifications.get('right_leg_swing', 0)
        torso_lean = modifications.get('torso_lean', 0)
        torso_sway_ap = modifications.get('torso_sway_ap', 0)
        torso_sway_ml = modifications.get('torso_sway_ml', 0)
        head_bob = modifications.get('head_bob', 0)
        torso_rotation = modifications.get('torso_rotation', 0)
        
        # Head - subtle movement only
        silhouette['head'] = {
            'x': [x + torso_sway_ml * 0.3 for x in base['head']['x']],
            'y': [y + torso_lean * 0.2 + head_bob for y in base['head']['y']]
        }
        
        # Neck - follows head naturally
        silhouette['neck'] = {
            'x': [x + torso_sway_ml * 0.4 for x in base['neck']['x']],
            'y': [y + torso_lean * 0.3 for y in base['neck']['y']]
        }
        
        # Torso - minimal movement
        silhouette['torso'] = {
            'x': [x + torso_sway_ml * 0.5 for x in base['torso']['x']],
            'y': [y + torso_lean * 0.5 for y in base['torso']['y']]
        }
        
        # Left arm - natural swing motion
        silhouette['left_upper_arm'] = {
            'x': [x - left_arm_swing * 0.5 for x in base['left_upper_arm']['x']],  # Forward/back motion
            'y': [y + left_arm_swing * 0.2 for y in base['left_upper_arm']['y']]   # Up/down motion
        }
        silhouette['left_forearm'] = {
            'x': [x - left_arm_swing * 0.8 for x in base['left_forearm']['x']],
            'y': [y + left_arm_swing * 0.3 for y in base['left_forearm']['y']]
        }
        silhouette['left_hand'] = {
            'x': [x - left_arm_swing * 1.0 for x in base['left_hand']['x']],
            'y': [y + left_arm_swing * 0.4 for y in base['left_hand']['y']]
        }
        
        # Right arm - opposite swing motion
        silhouette['right_upper_arm'] = {
            'x': [x - right_arm_swing * 0.5 for x in base['right_upper_arm']['x']],
            'y': [y + right_arm_swing * 0.2 for y in base['right_upper_arm']['y']]
        }
        silhouette['right_forearm'] = {
            'x': [x - right_arm_swing * 0.8 for x in base['right_forearm']['x']],
            'y': [y + right_arm_swing * 0.3 for y in base['right_forearm']['y']]
        }
        silhouette['right_hand'] = {
            'x': [x - right_arm_swing * 1.0 for x in base['right_hand']['x']],
            'y': [y + right_arm_swing * 0.4 for y in base['right_hand']['y']]
        }
        
        # Legs - minimal movement, mostly in Y direction
        for leg_parts, leg_swing in [(['left_thigh', 'left_shin', 'left_foot'], left_leg_swing),
                                    (['right_thigh', 'right_shin', 'right_foot'], right_leg_swing)]:
            for part in leg_parts:
                silhouette[part] = {
                    'x': [x + leg_swing * 0.3 for x in base[part]['x']],  # Minimal side-to-side
                    'y': [y for y in base[part]['y']]  # Keep legs mostly stationary
                }
        
        return silhouette

# Initialize data loader and load all datasets - Comprehensive data integration
print("Initializing multi-dataset Parkinson's analysis...")
loader = ParkinsonDataLoader()
df_clean = loader.load_all_datasets()

# Initialize motion silhouette generator - FIXED motion visualization system
motion_generator = MotionSilhouetteGenerator()

if df_clean.empty:
    print("❌ No data loaded. Please check file paths.")
    df_clean = pd.DataFrame()
else:
    patients = sorted(df_clean['PATNO'].unique())
    print(f"✓ Successfully loaded data for {len(patients)} patients")

# Enhanced feature groups - Comprehensive multi-modal feature organization
FEATURE_GROUPS = {
    'Objective Motor': ['SP_U', 'ASA_U', 'ARM_ASYMMETRY', 'TOTAL_JERK', 'MOVEMENT_QUALITY'],
    'Clinical Scores': ['CLINICAL_MOTOR_SEVERITY', 'PATIENT_REPORTED_SEVERITY', 'AGE_ADJUSTED_SEVERITY'],
    'Bilateral Comparison': ['RA_AMP_U', 'LA_AMP_U', 'R_JERK_U', 'L_JERK_U', 'BILATERAL_COORDINATION'],
    'Composite Indices': ['OBJECTIVE_MOTOR_SCORE', 'SENSOR_CLINICAL_RATIO'],
    'Digital Biomarkers': ['SENSOR_MEAN', 'SENSOR_STD', 'SENSOR_COUNT']
}

# Enhanced feature labels - Clinical interpretation-focused labeling
FEATURE_LABELS = {
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
}

# Initialize the Dash App
app = dash.Dash(__name__)

# FIXED App Layout - Proper animation integration with enhanced motion silhouettes
app.layout = html.Div(style={'fontFamily': 'Arial', 'padding': '20px', 'backgroundColor': '#f8f9fa'}, children=[
    html.H1("Multi-Modal Parkinson's Disease Analysis with Motion Silhouettes", 
            style={'color': '#2c3e50', 'textAlign': 'center', 'marginBottom': '10px'}),
    html.P("Real-time patient motion visualization through anatomical silhouettes and multi-modal data integration",
           style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px', 'marginBottom': '30px'}),

    # Enhanced control panel - Multi-dimensional analysis controls with motion settings
    html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px', 'backgroundColor': 'white', 
                   'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}, children=[
        html.Div(style={'width': '200px'}, children=[
            html.Label("Select Patient:", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
            dcc.Dropdown(
                id='patient-dropdown',
                options=[{'label': f'Patient {p}', 'value': p} for p in patients] if not df_clean.empty else [],
                placeholder="Select a patient...",
                style={'marginTop': '5px'}
            )
        ]),
        html.Div(style={'width': '200px'}, children=[
            html.Label("Motion Test Type:", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
            dcc.Dropdown(
                id='motion-test-dropdown',
                options=[
                    {'label': 'Gait/Walking Test', 'value': 'gait'},
                    {'label': 'TUG Test (Timed Up & Go)', 'value': 'tug'},
                    {'label': 'Postural Sway/Balance', 'value': 'balance'},
                    {'label': 'Free Motion', 'value': 'free'}
                ],
                value='gait',
                style={'marginTop': '5px'}
            )
        ]),
        html.Div(style={'width': '200px'}, children=[
            html.Label("Animation Speed:", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
            dcc.Slider(
                id='animation-speed-slider',
                min=0.1,
                max=3.0,
                step=0.1,
                value=1.0,
                marks={0.5: '0.5x', 1.0: '1.0x', 2.0: '2.0x', 3.0: '3.0x'},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ]),
        html.Div(style={'width': '200px'}, children=[
            html.Label("X-Axis Feature:", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
            dcc.Dropdown(
                id='x-axis-dropdown',
                options=[{'label': FEATURE_LABELS.get(feat, feat), 'value': feat} 
                        for feat in ['CLINICAL_MOTOR_SEVERITY', 'OBJECTIVE_MOTOR_SCORE', 'SP_U', 'ASA_U', 'SENSOR_MEAN']],
                value='CLINICAL_MOTOR_SEVERITY',
                style={'marginTop': '5px'}
            )
        ]),
        html.Div(style={'width': '200px'}, children=[
            html.Label("Y-Axis Feature:", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
            dcc.Dropdown(
                id='y-axis-dropdown',
                options=[{'label': FEATURE_LABELS.get(feat, feat), 'value': feat} 
                        for feat in ['MOVEMENT_QUALITY', 'OBJECTIVE_MOTOR_SCORE', 'SP_U', 'BILATERAL_COORDINATION', 'SENSOR_CLINICAL_RATIO']],
                value='MOVEMENT_QUALITY',
                style={'marginTop': '5px'}
            )
        ])
    ]),

    # FIXED: Motion Silhouette Panel - Real-time anatomical motion visualization with proper updates
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '2fr 1fr', 'gap': '20px', 'marginBottom': '30px'}, children=[
        # Main correlation plot
        dcc.Graph(id='main-correlation-plot', style={'height': '60vh'}),
        
        # Motion silhouette visualization - FIXED to update with animation
        html.Div(style={'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '20px', 
                       'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}, children=[
            html.H3("Real-Time Motion Silhouette", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            dcc.Graph(id='motion-silhouette-plot', style={'height': '45vh'}),
            html.Div(id='motion-metrics-display', style={'marginTop': '10px', 'fontSize': '12px'})
        ])
    ]),
    
    # Motion Analysis Dashboard - Comprehensive motion-specific visualizations
    html.H2("Dynamic Motion Analysis Dashboard", 
            style={'color': '#2c3e50', 'marginTop': '40px', 'marginBottom': '20px'}),
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '20px', 'marginBottom': '30px'}, children=[
        dcc.Graph(id='bilateral-asymmetry-motion', style={'height': '40vh'}),
        dcc.Graph(id='gait-cycle-analysis', style={'height': '40vh'}),
        dcc.Graph(id='motion-quality-assessment', style={'height': '40vh'})
    ]),
    
    # Animation Control - FIXED real-time motion animation controls
    html.Div(style={'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '20px', 
                   'marginTop': '30px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}, children=[
        html.H3("Motion Animation Controls", style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Div(style={'display': 'flex', 'gap': '20px', 'alignItems': 'center'}, children=[
            html.Button("Play", id='play-button', n_clicks=0, 
                       style={'padding': '10px 20px', 'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
            html.Button("Pause", id='pause-button', n_clicks=0,
                       style={'padding': '10px 20px', 'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
            html.Button("Reset", id='reset-button', n_clicks=0,
                       style={'padding': '10px 20px', 'backgroundColor': '#95a5a6', 'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
            html.Div(id='animation-status', style={'marginLeft': '20px', 'color': '#2c3e50'})
        ])
    ]),
    
    # FIXED Animation timer for real-time updates - Proper continuous motion animation
    dcc.Interval(
        id='animation-interval',
        interval=100,  # Faster for smoother animation
        n_intervals=0,
        disabled=False  # <-- Enable by default
    ),
    
    # Store animation state with proper initialization
    dcc.Store(id='animation-state', data={'playing': True, 'time_phase': 0, 'speed': 1.0}),
    # Store for selected patient data to avoid constant lookups
    dcc.Store(id='patient-data-store')
])


def create_motion_silhouette_plot(patient_data, motion_test, time_phase):
    """FIXED: Create real-time motion silhouette with proper error handling"""
    
    try:
        # Generate motion frame using patient's actual data
        silhouette_frame = motion_generator.generate_motion_frame(patient_data, motion_test, time_phase)
        
        fig = go.Figure()
        
        # Color scheme for different body parts
        body_colors = {
            'head': '#3498db', 'neck': '#2980b9', 'torso': '#2c3e50', 
            'left_upper_arm': '#e74c3c', 'left_forearm': '#c0392b', 'left_hand': '#a93226',
            'right_upper_arm': '#27ae60', 'right_forearm': '#229954', 'right_hand': '#1e8449',
            'left_thigh': '#f39c12', 'left_shin': '#e67e22', 'left_foot': '#d35400',
            'right_thigh': '#9b59b6', 'right_shin': '#8e44ad', 'right_foot': '#7d3c98'
        }
        
        # Draw each body part with error checking
        for part_name, coordinates in silhouette_frame.items():
            if coordinates and 'x' in coordinates and 'y' in coordinates:
                if coordinates['x'] and coordinates['y']:  # Check for valid coordinates
                    try:
                        fig.add_trace(go.Scatter(
                            x=coordinates['x'] + [coordinates['x'][0]],  # Close the shape
                            y=coordinates['y'] + [coordinates['y'][0]], 
                            fill='toself',
                            fillcolor=body_colors.get(part_name, '#95a5a6'),
                            line=dict(color='black', width=1),
                            name=part_name.replace('_', ' ').title(),
                            hoverinfo='name',
                            opacity=0.8
                        ))
                    except Exception as e:
                        print(f"Error adding trace for {part_name}: {e}")
                        continue
        
        # Add motion indicators with error checking
        if patient_data:
            try:
                # Add asymmetry indicator
                asymmetry = patient_data.get('ASA_U', 0)
                if asymmetry and not pd.isna(asymmetry) and asymmetry > 0.5:
                    fig.add_annotation(
                        x=-2, y=8, text=f"⚠️ High Asymmetry: {asymmetry:.2f}",
                        showarrow=False, bgcolor="rgba(231,76,60,0.9)",
                        bordercolor="red", font=dict(color="white", size=10), borderwidth=1
                    )
                
                # Add speed indicator
                speed = patient_data.get('SP_U', 1.0)
                if speed and not pd.isna(speed):
                    if speed < 0.8:
                        speed_status, status_color = "🐌 Slow Gait", "rgba(231,76,60,0.9)"
                    elif speed > 1.2:
                        speed_status, status_color = "🏃 Fast Gait", "rgba(52,152,219,0.9)"
                    else:
                        speed_status, status_color = "🚶 Normal Gait", "rgba(46,204,113,0.9)"
                    
                    fig.add_annotation(
                        x=2, y=8, text=f"{speed_status}<br>{speed:.2f} m/s",
                        showarrow=False, bgcolor=status_color, bordercolor="gray",
                        font=dict(color="white", size=10), borderwidth=1
                    )
            except Exception as e:
                print(f"Error adding annotations: {e}")
        
        # Configure layout
        fig.update_layout(
            title=f"Motion Silhouette - {motion_test.replace('_', ' ').title()} Test<br><sub>Phase: {time_phase:.1f} rad</sub>",
            xaxis=dict(range=[-3, 3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-4, 9], showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=10, r=10, t=50, b=10), height=500
        )
        
        motion_metrics = create_motion_metrics_display(patient_data, motion_test)
        return fig, motion_metrics
        
    except Exception as e:
        print(f"Error in create_motion_silhouette_plot: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error generating motion silhouette: {str(e)}", showarrow=False)
        fig.update_layout(title="Motion Silhouette - Error", height=500)
        error_metrics = html.Div([html.P("Error generating motion metrics", style={'color': 'red'})])
        return fig, error_metrics

def create_motion_metrics_display(patient_data, motion_test):
    """FIXED: Create motion metrics display using actual patient data"""
    if not patient_data:
        return html.Div([
            html.P("👤 No patient selected", style={'color': '#7f8c8d', 'fontStyle': 'italic'}),
            html.P("Select a patient to see personalized motion metrics", style={'color': '#bdc3c7', 'fontSize': '11px'})
        ])
    
    metrics = []
    
    if 'SP_U' in patient_data and not pd.isna(patient_data['SP_U']):
        speed = patient_data['SP_U']
        if speed < 0.8: speed_color, speed_icon = '#e74c3c', '🐌'
        elif speed > 1.2: speed_color, speed_icon = '#3498db', '🏃'
        else: speed_color, speed_icon = '#27ae60', '🚶'
        metrics.append(html.Div([
            html.Span(f"{speed_icon} Gait Speed: ", style={'fontWeight': 'bold'}),
            html.Span(f"{speed:.2f} m/s", style={'color': speed_color, 'fontWeight': 'bold'})
        ], style={'marginBottom': '5px'}))
    
    if 'ASA_U' in patient_data and not pd.isna(patient_data['ASA_U']):
        asa = patient_data['ASA_U']
        if asa < 0.2: asa_color, asa_icon = '#27ae60', '✅'
        elif asa < 0.5: asa_color, asa_icon = '#f39c12', '⚠️'
        else: asa_color, asa_icon = '#e74c3c', '🚨'
        metrics.append(html.Div([
            html.Span(f"{asa_icon} Arm Asymmetry: ", style={'fontWeight': 'bold'}),
            html.Span(f"{asa:.3f}", style={'color': asa_color, 'fontWeight': 'bold'})
        ], style={'marginBottom': '5px'}))

    if not metrics:
        return html.Div([
            html.P("📊 Limited motion data available", style={'color': '#f39c12', 'fontStyle': 'italic'}),
            html.P("Some metrics may not be available for this patient", style={'color': '#bdc3c7', 'fontSize': '11px'})
        ])
    
    return html.Div(metrics)

def create_enhanced_correlation_plot(df, x_feature, y_feature, selected_patient):
    """Enhanced correlation plot with motion context"""
    valid_data = df.dropna(subset=[x_feature, y_feature])
    if valid_data.empty:
        return go.Figure().add_annotation(text="No valid data for selected features", showarrow=False)

    fig = px.scatter(
        valid_data, x=x_feature, y=y_feature, color='COHORT_NAME',
        title=f'Analysis: {FEATURE_LABELS.get(y_feature, y_feature)} vs. {FEATURE_LABELS.get(x_feature, x_feature)}'
    )
    
    if selected_patient and selected_patient in valid_data['PATNO'].values:
        patient_data = valid_data[valid_data['PATNO'] == selected_patient]
        fig.add_trace(go.Scatter(
            x=patient_data[x_feature], y=patient_data[y_feature], mode='markers',
            marker=dict(size=20, color='red', symbol='star', line=dict(width=3, color='black')),
            name=f'Patient {selected_patient}'
        ))
    
    fig.update_layout(
        xaxis_title=FEATURE_LABELS.get(x_feature, x_feature),
        yaxis_title=FEATURE_LABELS.get(y_feature, y_feature),
        template='plotly_white'
    )
    return fig

def create_bilateral_asymmetry_motion(df, selected_patient):
    """Create bilateral asymmetry visualization"""
    if 'RA_AMP_U' not in df.columns or 'LA_AMP_U' not in df.columns:
        return go.Figure().add_annotation(text="Bilateral arm data not available", showarrow=False)
    
    valid_data = df.dropna(subset=['RA_AMP_U', 'LA_AMP_U'])
    fig = px.scatter(valid_data, x='RA_AMP_U', y='LA_AMP_U', color='COHORT_NAME',
                     title='Bilateral Arm Movement Asymmetry')

    max_val = max(valid_data['RA_AMP_U'].max(), valid_data['LA_AMP_U'].max())
    fig.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val, 
                  line=dict(dash='dash', color='gray'))

    if selected_patient and selected_patient in valid_data['PATNO'].values:
        patient_data = valid_data[valid_data['PATNO'] == selected_patient]
        fig.add_trace(go.Scatter(
            x=patient_data['RA_AMP_U'], y=patient_data['LA_AMP_U'], mode='markers',
            marker=dict(size=20, color='red', symbol='star', line=dict(width=3, color='black')),
            name=f'Patient {selected_patient}'
        ))
        
    fig.update_layout(xaxis_title='Right Arm Amplitude', yaxis_title='Left Arm Amplitude', template='plotly_white')
    return fig

def create_gait_cycle_analysis(df, selected_patient, time_phase):
    """Create gait cycle analysis"""
    fig = go.Figure()
    if not selected_patient or selected_patient not in df['PATNO'].values:
        fig.add_annotation(text="Select a patient to see gait cycle", showarrow=False)
        return fig

    patient_data = df[df['PATNO'] == selected_patient].iloc[0]
    left_amp = patient_data.get('LA_AMP_U', 30)
    right_amp = patient_data.get('RA_AMP_U', 30)
    time_points = np.linspace(0, 2 * math.pi, 100)
    left_swing = (left_amp / 50.0) * np.sin(time_points)
    right_swing = (right_amp / 50.0) * np.sin(time_points + math.pi)

    fig.add_trace(go.Scatter(x=time_points, y=left_swing, mode='lines', name='Left Arm Swing'))
    fig.add_trace(go.Scatter(x=time_points, y=right_swing, mode='lines', name='Right Arm Swing'))
    fig.add_vline(x=time_phase, line_dash="dash", line_color="red", annotation_text="Current Phase")
    
    fig.update_layout(title=f'Gait Cycle - Patient {selected_patient}',
                      xaxis_title='Phase (radians)', yaxis_title='Amplitude', template='plotly_white')
    return fig

def create_motion_quality_assessment(df, selected_patient):
    """Create motion quality radar chart"""
    fig = go.Figure()
    if not selected_patient or selected_patient not in df['PATNO'].values:
        fig.add_annotation(text="Select a patient to see quality assessment", showarrow=False)
        return fig

    patient_data = df[df['PATNO'] == selected_patient].iloc[0]
    metrics = {
        'Movement Quality': min(patient_data.get('MOVEMENT_QUALITY', 0) / 20, 1.0),
        'Coordination': patient_data.get('BILATERAL_COORDINATION', 0),
        'Symmetry': max(0, 1 - (patient_data.get('ASA_U', 2.0) / 2.0))
    }
    
    fig.add_trace(go.Scatterpolar(
        r=list(metrics.values()),
        theta=list(metrics.keys()),
        fill='toself', name='Patient Quality'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      title=f'Motion Quality - Patient {selected_patient}', template='plotly_white')
    return fig

# Callback 1: Update the patient data store when the dropdown changes
@app.callback(
    Output('patient-data-store', 'data'),
    Input('patient-dropdown', 'value')
)
def update_patient_store(selected_patient):
    """
    Fetches the selected patient's data and stores it.
    This avoids expensive lookups in the high-frequency animation callback.
    """
    if selected_patient and not df_clean.empty and selected_patient in df_clean['PATNO'].values:
        patient_row = df_clean[df_clean['PATNO'] == selected_patient].iloc[0]
        return patient_row.to_dict()
    # Return average data if no patient is selected
    if not df_clean.empty:
        return {
            'LA_AMP_U': df_clean['LA_AMP_U'].mean(),
            'RA_AMP_U': df_clean['RA_AMP_U'].mean(),
            'SP_U': df_clean['SP_U'].mean(),
            'ASA_U': df_clean['ASA_U'].mean(),
            'L_JERK_U': df_clean.get('L_JERK_U', pd.Series([0.02])).mean(),
            'R_JERK_U': df_clean.get('R_JERK_U', pd.Series([0.02])).mean(),
        }
    return {}

# Callback 2: Control the animation state (Play, Pause, Reset, Speed)
@app.callback(
    [Output('animation-interval', 'disabled'),
     Output('animation-state', 'data'),
     Output('animation-status', 'children')],
    [Input('play-button', 'n_clicks'),
     Input('pause-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('animation-speed-slider', 'value')],
    [State('animation-state', 'data')],
    prevent_initial_call=False
)
def control_animation(play_clicks, pause_clicks, reset_clicks, speed, current_state):
    """
    Manages the master animation state based on user controls.
    Updates playing status, speed, and resets the time phase.
    """
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial'

    state = current_state or {'playing': True, 'time_phase': 0, 'speed': 1.0}
    state['speed'] = speed

    if triggered_id == 'play-button':
        state['playing'] = True
        return False, state, "🎬 Animation playing"
    elif triggered_id == 'pause-button':
        state['playing'] = False
        return True, state, "⏸️ Animation paused"
    elif triggered_id == 'reset-button':
        state['time_phase'] = 0
        state['playing'] = True
        return False, state, "🔄 Animation reset and playing"
    
    # Handle speed changes or initial load
    if state['playing']:
        return False, state, "🎬 Animation playing"
    else:
        return True, state, "⏸️ Animation paused"

# Callback 3: The "Ticker" - updates the time phase on every interval
@app.callback(
    Output('animation-state', 'data', allow_duplicate=True),
    Input('animation-interval', 'n_intervals'),
    State('animation-state', 'data'),
    prevent_initial_call=True,
)
def update_animation_phase(n_intervals, current_state):
    """
    This is the lightweight animation "engine". It fires every 100ms
    and its ONLY job is to increment the time_phase in the state store.
    """
    if current_state and current_state.get('playing'):
        # Increment time_phase based on speed
        current_state['time_phase'] = (current_state['time_phase'] + current_state['speed'] * 0.1) % (2 * math.pi)
        return current_state
    return dash.no_update

# Callback 4: The "Renderer" - creates the silhouette when the state changes
@app.callback(
    [Output('motion-silhouette-plot', 'figure'),
     Output('motion-metrics-display', 'children')],
    [Input('patient-data-store', 'data'),
     Input('animation-state', 'data'),
     Input('motion-test-dropdown', 'value')]
)
def update_motion_silhouette(patient_data, animation_state, motion_test):
    """
    This is the main rendering callback. It's triggered only when the
    underlying data (patient or animation frame) changes. It is NOT
    triggered directly by the high-frequency interval.
    """
    if not patient_data or not animation_state:
        return go.Figure(), "Select a patient to begin."

    time_phase = animation_state.get('time_phase', 0)
    
    try:
        fig_silhouette, motion_metrics = create_motion_silhouette_plot(patient_data, motion_test or 'gait', time_phase)
        return fig_silhouette, motion_metrics
    except Exception as e:
        print(f"Error in motion silhouette generation: {e}")
        empty_fig = go.Figure().add_annotation(text=f"Error: {e}", showarrow=False)
        error_msg = html.Div(f"Error: {e}", style={'color': 'red'})
        return empty_fig, error_msg

# Callback 5: Update all other (static) visualizations
@app.callback(
    [Output('main-correlation-plot', 'figure'),
     Output('bilateral-asymmetry-motion', 'figure'),
     Output('motion-quality-assessment', 'figure'),
     Output('gait-cycle-analysis', 'figure')],
    [Input('patient-dropdown', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('animation-state', 'data')] # Use animation state for gait cycle phase
)
def update_all_visualizations(selected_patient, x_feature, y_feature, animation_state):
    """
    Consolidated callback to update all non-silhouette plots.
    """
    if df_clean.empty:
        empty_fig = go.Figure().add_annotation(text="No data available", showarrow=False)
        return [empty_fig] * 4

    time_phase = animation_state.get('time_phase', 0)

    fig_main = create_enhanced_correlation_plot(df_clean, x_feature, y_feature, selected_patient)
    fig_bilateral = create_bilateral_asymmetry_motion(df_clean, selected_patient)
    fig_motion_quality = create_motion_quality_assessment(df_clean, selected_patient)
    fig_gait_cycle = create_gait_cycle_analysis(df_clean, selected_patient, time_phase)
    
    return fig_main, fig_bilateral, fig_motion_quality, fig_gait_cycle

# Run the App
if __name__ == '__main__':
    print("Starting FIXED Enhanced Multi-Modal Parkinson's Dashboard...")
    app.run(debug=True) 