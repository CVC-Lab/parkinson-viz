# enhanced_motion_data_inspector.py - Detailed Motion Data Analysis for Silhouette Visualization

import pandas as pd
import numpy as np
import os
import glob

class MotionDataInspector:
    def __init__(self, base_path='.'):
        self.base_path = base_path
        self.motion_datasets = {}
        
    def inspect_motion_datasets(self):
        """Deep dive into motion-specific datasets for silhouette visualization"""
        
        print("="*80)
        print("MOTION DATA ANALYSIS FOR SILHOUETTE VISUALIZATION")
        print("="*80)
        
        # 1. Analyze Gait Data for Walking Motion
        self.analyze_gait_data()
        
        # 2. Analyze Digital Sensor Data for Real-time Motion
        self.analyze_digital_sensor_data()
        
        # 3. Examine Individual Patient Sensor Files
        self.analyze_patient_sensor_files()
        
        # 4. Look for Test-Specific Motion Data
        self.analyze_test_specific_data()
        
        # 5. Generate Motion Visualization Strategy
        self.generate_motion_visualization_strategy()
    
    def analyze_gait_data(self):
        """Analyze gait data for walking motion patterns"""
        print(f"\n{'='*20} GAIT MOTION ANALYSIS {'='*20}")
        
        try:
            gait_path = os.path.join(self.base_path, 'Motor_Assessments', 'Gait_Data___Arm_swing_06Jan2025.csv')
            gait_df = pd.read_csv(gait_path)
            
            print(f"Gait Dataset Shape: {gait_df.shape}")
            
            # Identify motion-related columns
            motion_columns = {
                'Speed/Velocity': [col for col in gait_df.columns if any(x in col.upper() for x in ['SP_', 'VEL', 'SPEED'])],
                'Arm Movement': [col for col in gait_df.columns if any(x in col.upper() for x in ['ARM', 'RA_', 'LA_', 'AMP'])],
                'Gait Characteristics': [col for col in gait_df.columns if any(x in col.upper() for x in ['GAIT', 'STEP', 'CAD', 'STR'])],
                'Asymmetry/Balance': [col for col in gait_df.columns if any(x in col.upper() for x in ['ASY', 'SYM', 'BAL'])],
                'Jerk/Smoothness': [col for col in gait_df.columns if any(x in col.upper() for x in ['JERK', 'SMOOTH'])],
                'TUG Test Data': [col for col in gait_df.columns if 'TUG' in col.upper()]
            }
            
            for category, columns in motion_columns.items():
                if columns:
                    print(f"\n{category} ({len(columns)} columns):")
                    for col in columns[:8]:  # Show first 8 columns
                        sample_data = gait_df[col].dropna()
                        if len(sample_data) > 0:
                            print(f"  • {col}: Range {sample_data.min():.3f} to {sample_data.max():.3f}, Mean {sample_data.mean():.3f}")
                    if len(columns) > 8:
                        print(f"  ... and {len(columns) - 8} more columns")
            
            # Look for bilateral comparisons (left vs right)
            bilateral_pairs = self.find_bilateral_pairs(gait_df.columns)
            if bilateral_pairs:
                print(f"\nBilateral Motion Pairs ({len(bilateral_pairs)} pairs):")
                for left_col, right_col in bilateral_pairs[:5]:
                    print(f"  • {left_col} <-> {right_col}")
            
            # Check for time-series indicators
            time_indicators = [col for col in gait_df.columns if any(x in col.upper() for x in ['TIME', 'DUR', 'FREQ'])]
            if time_indicators:
                print(f"\nTemporal Indicators: {time_indicators}")
            
            self.motion_datasets['gait'] = {
                'dataframe': gait_df,
                'motion_columns': motion_columns,
                'bilateral_pairs': bilateral_pairs,
                'time_indicators': time_indicators
            }
            
        except Exception as e:
            print(f"❌ Error analyzing gait data: {e}")
    
    def analyze_digital_sensor_data(self):
        """Analyze digital sensor data for high-resolution motion"""
        print(f"\n{'='*20} DIGITAL SENSOR MOTION ANALYSIS {'='*20}")
        
        try:
            sensor_path = os.path.join(self.base_path, 'Digital_Sensor', 'Roche_PD_Monitoring_App_v2_data_06Jan2025.csv')
            sensor_df = pd.read_csv(sensor_path)
            
            print(f"Digital Sensor Dataset Shape: {sensor_df.shape}")
            
            # Analyze test categories
            test_categories = sensor_df['QRSSCAT'].value_counts()
            print(f"\nTest Categories:")
            for category, count in test_categories.head(10).items():
                print(f"  • {category}: {count} measurements")
            
            # Analyze test types
            test_types = sensor_df['QRSTEST'].value_counts()
            print(f"\nSpecific Test Types:")
            for test_type, count in test_types.head(15).items():
                print(f"  • {test_type}: {count} measurements")
            
            # Look for motion-specific tests
            motion_tests = sensor_df[sensor_df['QRSTEST'].str.contains('gait|walk|balance|sway|tap|tremor|move', case=False, na=False)]
            if len(motion_tests) > 0:
                print(f"\nMotion-Related Tests ({len(motion_tests)} records):")
                motion_test_types = motion_tests[['QRSTEST', 'QRSSCAT']].drop_duplicates()
                for _, row in motion_test_types.head(10).iterrows():
                    print(f"  • {row['QRSTEST']} ({row['QRSSCAT']})")
            
            # Analyze result values
            numeric_results = sensor_df['QRSRESN'].dropna()
            if len(numeric_results) > 0:
                print(f"\nNumeric Results Range: {numeric_results.min():.3f} to {numeric_results.max():.3f}")
                print(f"Result Distribution Quartiles: {numeric_results.quantile([0.25, 0.5, 0.75]).values}")
            
            # Check for time series data
            time_cols = [col for col in sensor_df.columns if any(x in col.upper() for x in ['TIME', 'DTM', 'DTC'])]
            print(f"\nTime Columns: {time_cols}")
            
            # Sample patient analysis
            sample_patients = sensor_df['PATNO'].unique()[:3]
            print(f"\nSample Patient Motion Patterns:")
            for patient in sample_patients:
                patient_data = sensor_df[sensor_df['PATNO'] == patient]
                unique_tests = patient_data['QRSTEST'].nunique()
                total_measurements = len(patient_data)
                print(f"  • Patient {patient}: {unique_tests} test types, {total_measurements} measurements")
            
            self.motion_datasets['digital_sensor'] = {
                'dataframe': sensor_df,
                'test_categories': test_categories,
                'test_types': test_types,
                'motion_tests': motion_tests,
                'time_columns': time_cols
            }
            
        except Exception as e:
            print(f"❌ Error analyzing digital sensor data: {e}")
    
    def analyze_patient_sensor_files(self):
        """Analyze individual patient sensor files for detailed motion data"""
        print(f"\n{'='*20} INDIVIDUAL PATIENT SENSOR FILES {'='*20}")
        
        sensor_dir = os.path.join(self.base_path, 'Digital_Sensor')
        patient_files = glob.glob(os.path.join(sensor_dir, "Patient-*.xlsx"))
        
        if len(patient_files) == 0:
            print("❌ No individual patient sensor files found")
            return
        
        print(f"Found {len(patient_files)} patient sensor files")
        
        # Analyze first few files in detail
        detailed_analysis = {}
        for i, file_path in enumerate(patient_files[:3]):
            try:
                filename = os.path.basename(file_path)
                patient_id = filename.split('-')[1].split('.')[0]
                print(f"\n--- Detailed Analysis: Patient {patient_id} ---")
                
                # Load Excel file
                xl_file = pd.ExcelFile(file_path)
                print(f"Sheets: {xl_file.sheet_names}")
                
                # Analyze main sheet
                df = pd.read_excel(file_path, sheet_name=0)
                print(f"Shape: {df.shape}")
                
                # Look for motion-specific columns
                motion_indicators = [col for col in df.columns if any(x in col.upper() for x in 
                                   ['ACCEL', 'GYRO', 'MAGNET', 'ORIENT', 'POSITION', 'VELOCITY', 'DISPLACEMENT'])]
                if motion_indicators:
                    print(f"Potential Motion Sensors: {motion_indicators}")
                
                # Analyze test types in this patient
                if 'QRSTEST' in df.columns:
                    patient_tests = df['QRSTEST'].value_counts()
                    print(f"Patient's Test Types:")
                    for test, count in patient_tests.head(8).items():
                        print(f"  • {test}: {count}")
                
                # Look for time-series structure
                if 'QRSDTM' in df.columns or 'QRSDTM_TIME' in df.columns:
                    time_span = self.analyze_time_span(df)
                    print(f"Time Span: {time_span}")
                
                # Check for continuous measurements vs discrete events
                if 'QRSRESN' in df.columns:
                    results = df['QRSRESN'].dropna()
                    if len(results) > 0:
                        print(f"Measurement Range: {results.min():.3f} to {results.max():.3f}")
                        
                        # Look for patterns that could indicate motion cycles
                        if len(results) > 10:
                            # Simple pattern detection
                            diff_vals = np.diff(results)
                            sign_changes = np.sum(np.diff(np.sign(diff_vals)) != 0)
                            print(f"Potential Motion Cycles (sign changes): {sign_changes}")
                
                detailed_analysis[patient_id] = {
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'motion_indicators': motion_indicators,
                    'sample_data': df.head(3) if len(df) > 0 else None
                }
                
            except Exception as e:
                print(f"❌ Error analyzing {filename}: {e}")
        
        self.motion_datasets['patient_files'] = detailed_analysis
    
    def analyze_test_specific_data(self):
        """Identify specific test types that could support silhouette animation"""
        print(f"\n{'='*20} TEST-SPECIFIC MOTION ANALYSIS {'='*20}")
        
        # From gait data - look for specific test protocols
        if 'gait' in self.motion_datasets:
            gait_df = self.motion_datasets['gait']['dataframe']
            
            # TUG test analysis
            tug_columns = [col for col in gait_df.columns if 'TUG' in col.upper()]
            if tug_columns:
                print(f"\nTimed Up and Go (TUG) Test Data:")
                print(f"Available TUG metrics: {len(tug_columns)}")
                
                # Analyze TUG phases
                tug_phases = {}
                for col in tug_columns:
                    if 'DUR' in col.upper():
                        phase = col.replace('TUG1_', '').replace('TUG2_', '')
                        tug_phases[phase] = col
                
                print(f"TUG Motion Phases: {list(tug_phases.keys())}")
                
                # Sample TUG data
                tug_sample = gait_df[tug_columns].dropna().head(3)
                if len(tug_sample) > 0:
                    print(f"Sample TUG measurements:")
                    print(tug_sample.to_string())
        
        # From digital sensor data - categorize by motion type
        if 'digital_sensor' in self.motion_datasets:
            sensor_df = self.motion_datasets['digital_sensor']['dataframe']
            
            # Categorize tests by motion type
            motion_categories = {
                'Walking/Gait': ['walk', 'gait', 'step'],
                'Balance/Postural': ['balance', 'sway', 'postur', 'stand'],
                'Tremor': ['tremor', 'shake', 'oscillat'],
                'Finger Tapping': ['tap', 'finger', 'dexterity'],
                'Voice/Speech': ['voice', 'speech', 'phonat'],
                'Cognitive': ['cognitive', 'memory', 'attention']
            }
            
            print(f"\nMotion Test Categorization:")
            for category, keywords in motion_categories.items():
                matching_tests = sensor_df[sensor_df['QRSTEST'].str.contains('|'.join(keywords), case=False, na=False)]
                if len(matching_tests) > 0:
                    unique_tests = matching_tests['QRSTEST'].unique()
                    print(f"\n{category} ({len(matching_tests)} measurements):")
                    for test in unique_tests[:5]:
                        count = len(matching_tests[matching_tests['QRSTEST'] == test])
                        print(f"  • {test}: {count} measurements")
    
    def find_bilateral_pairs(self, columns):
        """Find left-right paired measurements"""
        bilateral_pairs = []
        
        # Common left-right prefixes
        left_prefixes = ['L_', 'LA_', 'LEFT_', 'L']
        right_prefixes = ['R_', 'RA_', 'RIGHT_', 'R']
        
        for col in columns:
            for left_prefix in left_prefixes:
                if col.startswith(left_prefix):
                    # Look for corresponding right column
                    for right_prefix in right_prefixes:
                        right_col = col.replace(left_prefix, right_prefix, 1)
                        if right_col in columns and right_col != col:
                            bilateral_pairs.append((col, right_col))
                            break
                    break
        
        return bilateral_pairs
    
    def analyze_time_span(self, df):
        """Analyze temporal characteristics of measurements"""
        time_info = {}
        
        if 'QRSDTM' in df.columns:
            # Convert to datetime if possible
            try:
                df['datetime'] = pd.to_datetime(df['QRSDTM'])
                time_span = df['datetime'].max() - df['datetime'].min()
                time_info['span'] = str(time_span)
                time_info['frequency'] = f"{len(df) / time_span.total_seconds():.3f} Hz" if time_span.total_seconds() > 0 else "N/A"
            except:
                time_info['span'] = "Unable to parse dates"
        
        return time_info
    
    def generate_motion_visualization_strategy(self):
        """Generate strategy for implementing silhouette-based motion visualization"""
        print(f"\n{'='*20} MOTION VISUALIZATION STRATEGY {'='*20}")
        
        strategy = {
            'Data Sources': {
                'Primary': 'Gait data (arm swing, speed, asymmetry)',
                'Secondary': 'Digital sensor measurements',
                'Temporal': 'TUG test phases, visit progression'
            },
            'Silhouette Components': {
                'Torso': 'Central body reference point',
                'Arms': 'Left/Right amplitude (LA_AMP_U, RA_AMP_U)',
                'Legs': 'Gait speed (SP_U), step characteristics',
                'Head': 'Postural stability indicators'
            },
            'Animation Parameters': {
                'Walking Cycle': 'Based on cadence (CAD_U) and step timing',
                'Arm Swing': 'Amplitude asymmetry (ASA_U) and smoothness (JERK)',
                'Balance': 'Postural sway from balance tests',
                'Speed': 'Gait speed variations (SP_U)'
            },
            'Test-Specific Animations': {
                'TUG Test': 'Standing -> Walking -> Turning -> Walking -> Sitting',
                'Gait Test': 'Continuous walking with arm swing',
                'Balance Test': 'Postural sway visualization',
                'Finger Tapping': 'Hand/finger movement patterns'
            }
        }
        
        print("\nVisualization Implementation Plan:")
        for category, details in strategy.items():
            print(f"\n{category}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    print(f"  • {key}: {value}")
            else:
                print(f"  {details}")
        
        # Key measurements for silhouette animation
        print(f"\nKey Measurements for Silhouette Animation:")
        key_measures = [
            "SP_U (Gait Speed) → Walking animation speed",
            "LA_AMP_U/RA_AMP_U (Arm Amplitudes) → Left/right arm swing",
            "ASA_U (Arm Swing Asymmetry) → Bilateral coordination",
            "R_JERK_U/L_JERK_U (Movement Smoothness) → Motion fluidity",
            "TUG1_DUR/TUG2_DUR → Test phase timing",
            "CAD_U (Cadence) → Step frequency",
            "Digital sensor QRSRESN → Real-time motion intensity"
        ]
        
        for measure in key_measures:
            print(f"  🎯 {measure}")
        
        # Data availability assessment
        print(f"\nData Availability Assessment:")
        if 'gait' in self.motion_datasets:
            gait_patients = len(self.motion_datasets['gait']['dataframe']['PATNO'].unique())
            print(f"  ✓ Gait data available for {gait_patients} patients")
        
        if 'digital_sensor' in self.motion_datasets:
            sensor_patients = len(self.motion_datasets['digital_sensor']['dataframe']['PATNO'].unique())
            print(f"  ✓ Digital sensor data for {sensor_patients} patients")
        
        if 'patient_files' in self.motion_datasets:
            detailed_patients = len(self.motion_datasets['patient_files'])
            print(f"  ✓ Detailed sensor files for {detailed_patients} patients")
        
        # Implementation recommendations
        print(f"\nImplementation Recommendations:")
        recommendations = [
            "Start with gait data silhouette (arms + walking motion)",
            "Use TUG test data for specific motion sequences",
            "Implement real-time sensor data overlay for motion intensity",
            "Add bilateral asymmetry visualization (color coding)",
            "Include temporal progression (visit-to-visit changes)",
            "Use SVG-based silhouette for smooth animations"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

# Run the motion data inspection
if __name__ == "__main__":
    inspector = MotionDataInspector()
    inspector.inspect_motion_datasets()