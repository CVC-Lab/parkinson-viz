# enhanced_inspect_data.py - Comprehensive Data Inspector for Parkinson's Study

import pandas as pd
import os
import glob
import numpy as np
from datetime import datetime

class ParkinsonDataInspector:
    def __init__(self, base_path='..'):
        self.base_path = base_path
        self.datasets = {}
        
    def inspect_all_datasets(self):
        """Inspect all datasets in the Parkinson's study directory"""
        
        # Define dataset categories and their file patterns
        dataset_info = {
            'Motor Assessments': {
                'path': 'Motor_Assessments',
                'files': ['Gait_Data___Arm_swing_06Jan2025.csv', 
                         'MDS-UPDRS_Part_III_06Jan2025.csv',
                         'MDS-UPDRS_Part_IV__Motor_Complications_06Jan2025.csv',
                         'MDS_UPDRS_Part_II__Patient_Questionnaire_06Jan2025.csv']
            },
            'Digital_Sensor': {
                'path': 'Digital_Sensor',
                'files': 'Patient-*.xlsx'  # Pattern for patient files
            },
            'Demographics': {
                'path': 'Subject_Characteristics',
                'files': ['Demographics_08Jan2025.csv', 'Participant_Status_08Jan2025.csv']
            },
            'Medical_History': {
                'path': 'Medical_History',
                'files': ['Features_of_Parkinsonism_06Jan2025.csv',
                         'Neurological_Exam_05Jan2025.csv',
                         'Other_Clinical_Features_06Jan2025.csv']
            }
        }
        
        print("="*80)
        print("PARKINSON'S DISEASE STUDY - COMPREHENSIVE DATA INSPECTION")
        print("="*80)
        
        for category, info in dataset_info.items():
            print(f"\n{'='*20} {category.upper()} {'='*20}")
            
            if category == 'Digital_Sensor':
                self.inspect_digital_sensor_files(info['path'])
            else:
                for filename in info['files']:
                    file_path = os.path.join(self.base_path, info['path'], filename)
                    self.inspect_csv_file(file_path, filename)
    
    def inspect_csv_file(self, file_path, filename):
        """Inspect a single CSV file"""
        try:
            print(f"\n--- {filename} ---")
            df = pd.read_csv(file_path)
            
            print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Show column names grouped by type
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            date_cols = [col for col in df.columns if 'DT' in col.upper() or 'DATE' in col.upper()]
            
            print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols) > 10 else ''}")
            print(f"Text columns ({len(text_cols)}): {text_cols[:10]}{'...' if len(text_cols) > 10 else ''}")
            if date_cols:
                print(f"Date columns: {date_cols}")
            
            # Show unique patients and events if available
            if 'PATNO' in df.columns:
                print(f"Unique patients: {df['PATNO'].nunique()}")
                print(f"Patient range: {df['PATNO'].min()} - {df['PATNO'].max()}")
            
            if 'EVENT_ID' in df.columns:
                print(f"Unique events: {df['EVENT_ID'].nunique()}")
                print(f"Event types: {df['EVENT_ID'].unique()[:10]}")
            
            # Show sample data
            print(f"\nFirst 3 rows:")
            print(df.head(3).to_string())
            
            # Show missing data summary
            missing_summary = df.isnull().sum()
            high_missing = missing_summary[missing_summary > len(df) * 0.5]
            if len(high_missing) > 0:
                print(f"\nColumns with >50% missing data: {len(high_missing)}")
                print(high_missing.head())
            
            # Store dataset info
            self.datasets[filename] = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'numeric_cols': numeric_cols,
                'text_cols': text_cols,
                'sample_data': df.head(3)
            }
            
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
        except Exception as e:
            print(f"❌ Error reading {filename}: {str(e)}")
    
    def inspect_digital_sensor_files(self, sensor_path):
        """Inspect digital sensor Excel files"""
        sensor_dir = os.path.join(self.base_path, sensor_path)
        
        # Get all patient Excel files
        patient_files = glob.glob(os.path.join(sensor_dir, "Patient-*.xlsx"))
        
        print(f"\n--- Digital Sensor Patient Files ---")
        print(f"Found {len(patient_files)} patient files")
        
        if len(patient_files) > 0:
            # Inspect first few files as examples
            for i, file_path in enumerate(patient_files[:3]):
                try:
                    filename = os.path.basename(file_path)
                    print(f"\n  Example {i+1}: {filename}")
                    
                    # Read Excel file (might have multiple sheets)
                    xl_file = pd.ExcelFile(file_path)
                    print(f"  Sheets: {xl_file.sheet_names}")
                    
                    # Inspect first sheet
                    df = pd.read_excel(file_path, sheet_name=0)
                    print(f"  Shape: {df.shape}")
                    print(f"  Columns: {df.columns.tolist()[:10]}{'...' if len(df.columns) > 10 else ''}")
                    
                    # Show time-series characteristics if available
                    time_cols = [col for col in df.columns if any(word in col.upper() for word in ['TIME', 'DATE', 'DTM'])]
                    if time_cols:
                        print(f"  Time columns: {time_cols}")
                    
                except Exception as e:
                    print(f"  ❌ Error reading {filename[:20]}...: {str(e)}")
        
        # Also check the main CSV file
        csv_path = os.path.join(sensor_dir, "Roche_PD_Monitoring_App_v2_data_06Jan2025.csv")
        if os.path.exists(csv_path):
            self.inspect_csv_file(csv_path, "Roche_PD_Monitoring_App_v2_data_06Jan2025.csv")
    
    def generate_integration_strategy(self):
        """Generate recommendations for data integration"""
        print(f"\n{'='*20} INTEGRATION STRATEGY {'='*20}")
        
        integration_plan = {
            'Primary Keys': {
                'PATNO': 'Patient identifier - available in most datasets',
                'EVENT_ID': 'Visit/event identifier - for longitudinal analysis',
                'INFODT/Date columns': 'Time dimension for temporal analysis'
            },
            'Clinical Hierarchy': {
                'Level 1 - Demographics': 'Basic patient characteristics (age, sex, diagnosis)',
                'Level 2 - Clinical Scores': 'UPDRS parts II, III, IV - standardized assessments',
                'Level 3 - Motor Assessments': 'Gait data, arm swing - objective measurements',
                'Level 4 - Digital Sensors': 'High-frequency sensor data - raw signals'
            },
            'Visualization Targets': {
                'Longitudinal Trends': 'Track clinical progression over visits',
                'Signal-to-Clinical Mapping': 'Connect sensor patterns to UPDRS scores',
                'Comparative Analysis': 'Compare cohorts, left vs right, on vs off medication',
                'Multi-modal Correlation': 'Relate different measurement types'
            }
        }
        
        for category, items in integration_plan.items():
            print(f"\n{category}:")
            if isinstance(items, dict):
                for key, desc in items.items():
                    print(f"  • {key}: {desc}")
            else:
                print(f"  {items}")
    
    def show_key_relationships(self):
        """Show key relationships between datasets"""
        print(f"\n{'='*20} KEY DATA RELATIONSHIPS {'='*20}")
        
        relationships = [
            "PATNO (Patient ID) → Links all datasets",
            "EVENT_ID → Connects longitudinal visits",
            "UPDRS Part III (NP3TOT) → Overall motor severity score",
            "Gait data (ASA_U, SP_U) → Objective movement measures",
            "Digital sensors → High-resolution behavioral data",
            "Demographics → Patient stratification variables"
        ]
        
        for rel in relationships:
            print(f"  🔗 {rel}")
            
        print(f"\nCLINICAL INTERPRETATION PIPELINE:")
        print(f"  Raw Sensors → Derived Features → Clinical Scores → Diagnosis/Progression")

# Run the inspection
if __name__ == "__main__":
    inspector = ParkinsonDataInspector()
    inspector.inspect_all_datasets()
    inspector.generate_integration_strategy()
    inspector.show_key_relationships()