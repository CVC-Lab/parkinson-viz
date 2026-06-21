"""Data pipeline: load / merge / clean the PPMI CSVs.

Importable WITHOUT building the Dash app, so convert_data_to_json.py can reuse
the loader without side effects."""
import os
import re
import numpy as np
import pandas as pd


def _normalize_event(ev):
    """Normalize PPMI visit IDs so joins match (e.g. 'v8', 'V8' -> 'V08')."""
    s = str(ev).strip().upper()
    m = re.match(r'^V0*(\d+)$', s)
    return 'V%02d' % int(m.group(1)) if m else s


_MONTHS = {m: i for i, m in enumerate(
    ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], start=1)}


def _month_ordinal(s):
    """Parse a PPMI 'Mon-YY' date to a year*12+month ordinal, or None."""
    m = re.match(r'^([A-Za-z]{3})-(\d{2})$', str(s).strip())
    if not m or m.group(1).upper() not in _MONTHS:
        return None
    return (2000 + int(m.group(2))) * 12 + _MONTHS[m.group(1).upper()]


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
            self.data['gait']['EVENT_ID'] = self.data['gait']['EVENT_ID'].map(_normalize_event)
            print(f"✓ Loaded gait data: {self.data['gait'].shape}")
        except Exception as e:
            print(f"Error loading gait data: {e}")
            self.data['gait'] = pd.DataFrame()
    
    def load_updrs_data(self):
        """Load UPDRS clinical scores - Added all UPDRS parts for comprehensive clinical assessment"""
        try:
            # UPDRS Part III (Motor examination)
            updrs3_path = os.path.join(self.base_path, 'Motor_Assessments', 'MDS-UPDRS_Part_III_06Jan2025.csv')
            updrs3 = pd.read_csv(updrs3_path)
            
            # Keep only essential columns to avoid memory issues - Selected key clinical indicators
            updrs3_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'PDSTATE', 'NP3TOT', 'NP3SPCH', 'NP3FACXP', 'NP3RIGN',
                          'NP3RIGRU', 'NP3RIGLU', 'NP3FTAPR', 'NP3FTAPL', 'NP3GAIT', 'NP3PSTBL',
                          'NP3BRADY', 'NP3PTRMR', 'NP3PTRML', 'NHY']
            self.data['updrs3'] = updrs3[updrs3_cols].copy()
            self.data['updrs3']['EVENT_ID'] = self.data['updrs3']['EVENT_ID'].map(_normalize_event)

            # UPDRS Part II (Patient questionnaire) - Added patient-reported outcomes
            updrs2_path = os.path.join(self.base_path, 'Motor_Assessments', 'MDS_UPDRS_Part_II__Patient_Questionnaire_06Jan2025.csv')
            updrs2 = pd.read_csv(updrs2_path)
            
            updrs2_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'NP2PTOT', 'NP2SPCH', 'NP2WALK', 'NP2TURN', 'NP2TRMR']
            self.data['updrs2'] = updrs2[updrs2_cols].copy()
            self.data['updrs2']['EVENT_ID'] = self.data['updrs2']['EVENT_ID'].map(_normalize_event)

            print(f"✓ Loaded UPDRS III: {self.data['updrs3'].shape}")
            print(f"✓ Loaded UPDRS II: {self.data['updrs2'].shape}")
            
        except Exception as e:
            print(f"Error loading UPDRS data: {e}")
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
            print(f"Error loading demographics: {e}")
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
            print(f"Error loading sensor data: {e}")
            self.data['sensors'] = pd.DataFrame()
            self.data['sensor_summary'] = pd.DataFrame()
    
    def merge_datasets(self):
        """Merge all datasets on PATNO and EVENT_ID - Comprehensive multi-modal data integration"""
        if self.data['gait'].empty:
            print("No gait data available for merging")
            self.data['merged'] = pd.DataFrame()
            return
        
        # Start with gait data as base - Using objective motor measurements as foundation
        merged = self.data['gait'].copy()
        
        # Add UPDRS scores - Integrated clinical severity assessments
        if not self.data['updrs3'].empty:
            merged = merged.merge(self.data['updrs3'], on=['PATNO', 'EVENT_ID'], how='left', suffixes=('', '_updrs3'))
            # Date-tolerance guard: a same-EVENT_ID join can still pair a gait visit with a
            # UPDRS exam years apart (PPMI reused some visit labels). If the gait date and the
            # UPDRS date differ by > 6 months, drop the UPDRS-III values for that row.
            if 'INFODT' in merged.columns and 'INFODT_updrs3' in merged.columns:
                g = merged['INFODT'].map(_month_ordinal)
                u = merged['INFODT_updrs3'].map(_month_ordinal)
                mismatch = g.notna() & u.notna() & ((g - u).abs() > 6)
                u3_cols = [c if c != 'INFODT' else 'INFODT_updrs3'
                           for c in self.data['updrs3'].columns if c not in ('PATNO', 'EVENT_ID')]
                for c in u3_cols:
                    if c in merged.columns:
                        merged.loc[mismatch, c] = np.nan
                if mismatch.any():
                    print(f"  ⚠ nulled {int(mismatch.sum())} UPDRS-III join(s) with >6mo gait/exam date gap")

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
        
        # Keep only rows with the gait measurements the app needs, THEN collapse to one
        # canonical record per participant (so dedupe chooses among usable rows, and
        # cohort-relative features are computed only over the kept participants).
        merged = merged.dropna(subset=['PATNO', 'ASA_U', 'SP_U'])
        merged = self.dedupe_participants(merged)

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
            merged['CLINICAL_MOTOR_SEVERITY'] = merged['NP3TOT']      # keep NaN — unknown != 0
        else:
            merged['CLINICAL_MOTOR_SEVERITY'] = np.nan

        if 'NP2PTOT' in merged.columns:
            merged['PATIENT_REPORTED_SEVERITY'] = merged['NP2PTOT']   # keep NaN — unknown != 0
        else:
            merged['PATIENT_REPORTED_SEVERITY'] = np.nan
        
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

    def dedupe_participants(self, df):
        """Collapse to one canonical record per participant.

        The raw merge fans out by visit and by ON/OFF medication state (PDSTATE),
        so a participant has several rows with different UPDRS-III totals. Keep one:
        prefer a row with a motor exam (NP3TOT present), then the latest visit, then
        OFF state, then the higher UPDRS-III (OFF ~ unmedicated). Makes participant
        selection deterministic and stops the cohort scatters double-counting people.
        """
        import re
        if df is None or df.empty or 'PATNO' not in df.columns:
            return df
        df = df.copy()

        def visit_rank(ev):
            s = str(ev).upper()
            if s == 'SC':
                return -2
            if s == 'BL':
                return 0
            m = re.search(r'(\d+)', s)
            return int(m.group(1)) if m else 0.5

        df['_has'] = df['NP3TOT'].notna().astype(int)
        df['_vr'] = df['EVENT_ID'].map(visit_rank)
        df['_off'] = (df['PDSTATE'].astype(str).str.upper() == 'OFF').astype(int) if 'PDSTATE' in df.columns else 0
        df['_np3'] = df['NP3TOT'].fillna(-1)

        df = df.sort_values(['PATNO', '_has', '_vr', '_off', '_np3'])
        df = df.drop_duplicates('PATNO', keep='last')
        return df.drop(columns=['_has', '_vr', '_off', '_np3']).reset_index(drop=True)
