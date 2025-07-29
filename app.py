# app.py - MULTI-DATASET ENHANCED PARKINSON'S VISUALIZATION

import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ParkinsonDataLoader:
    """Enhanced data loader for multi-modal Parkinson's datasets"""
    def __init__(self, base_path='..'):
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

# Initialize data loader and load all datasets - Comprehensive data integration
print("Initializing multi-dataset Parkinson's analysis...")
loader = ParkinsonDataLoader()
df_clean = loader.load_all_datasets()

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

# Enhanced App Layout - Comprehensive multi-modal visualization interface
app.layout = html.Div(style={'fontFamily': 'Arial', 'padding': '20px', 'backgroundColor': '#f8f9fa'}, children=[
    html.H1("Multi-Modal Parkinson's Disease Analysis Dashboard", 
            style={'color': '#2c3e50', 'textAlign': 'center', 'marginBottom': '10px'}),
    html.P("Connecting low-level sensor signals to clinical interpretations through multi-modal data integration",
           style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px', 'marginBottom': '30px'}),

    # Enhanced control panel - Multi-dimensional analysis controls
    html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px', 'backgroundColor': 'white', 
                   'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}, children=[
        html.Div(style={'width': '300px'}, children=[
            html.Label("Highlight Patient:", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
            dcc.Dropdown(
                id='patient-dropdown',
                options=[{'label': f'Patient {p}', 'value': p} for p in patients] if not df_clean.empty else [],
                placeholder="Select a patient to highlight...",
                style={'marginTop': '5px'}
            )
        ]),
        html.Div(style={'width': '300px'}, children=[
            html.Label("X-Axis Feature:", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
            dcc.Dropdown(
                id='x-axis-dropdown',
                options=[{'label': FEATURE_LABELS.get(feat, feat), 'value': feat} 
                        for feat in ['CLINICAL_MOTOR_SEVERITY', 'OBJECTIVE_MOTOR_SCORE', 'SP_U', 'ASA_U', 'SENSOR_MEAN']],
                value='CLINICAL_MOTOR_SEVERITY',
                style={'marginTop': '5px'}
            )
        ]),
        html.Div(style={'width': '300px'}, children=[
            html.Label("Y-Axis Feature:", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
            dcc.Dropdown(
                id='y-axis-dropdown',
                options=[{'label': FEATURE_LABELS.get(feat, feat), 'value': feat} 
                        for feat in ['MOVEMENT_QUALITY', 'OBJECTIVE_MOTOR_SCORE', 'SP_U', 'BILATERAL_COORDINATION', 'SENSOR_CLINICAL_RATIO']],
                value='MOVEMENT_QUALITY',
                style={'marginTop': '5px'}
            )
        ]),
        html.Div(style={'width': '200px'}, children=[
            html.Label("Analysis Mode:", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
            dcc.Dropdown(
                id='analysis-mode',
                options=[
                    {'label': 'Cross-Sectional', 'value': 'cross'},
                    {'label': 'Longitudinal', 'value': 'longitudinal'},
                    {'label': 'Multi-Modal', 'value': 'multimodal'}
                ],
                value='multimodal',
                style={'marginTop': '5px'}
            )
        ])
    ]),

    # Main visualization - Enhanced multi-modal correlation plot
    dcc.Graph(id='main-correlation-plot', style={'height': '70vh', 'marginBottom': '30px'}),
    
    # Multi-panel comparative analysis - Comprehensive comparative visualization
    html.H2("Multi-Modal Comparative Analysis", 
            style={'color': '#2c3e50', 'marginTop': '40px', 'marginBottom': '20px'}),
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginBottom': '30px'}, children=[
        dcc.Graph(id='clinical-objective-correlation', style={'height': '45vh'}),
        dcc.Graph(id='bilateral-asymmetry-analysis', style={'height': '45vh'})
    ]),
    
    # Lower panel - sensor-clinical integration - Digital biomarker integration
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginBottom': '30px'}, children=[
        dcc.Graph(id='sensor-clinical-mapping', style={'height': '45vh'}),
        dcc.Graph(id='cohort-progression-analysis', style={'height': '45vh'})
    ]),
    
    # Longitudinal trends - Enhanced time series with multi-modal integration
    html.H2("Longitudinal Multi-Modal Trends", 
            style={'color': '#2c3e50', 'marginTop': '40px', 'marginBottom': '20px'}),
    dcc.Graph(id='longitudinal-trends', style={'height': '55vh'})
])

# Enhanced callbacks - Comprehensive multi-modal visualization logic
@app.callback(
    [Output('main-correlation-plot', 'figure'),
     Output('clinical-objective-correlation', 'figure'),
     Output('bilateral-asymmetry-analysis', 'figure'),
     Output('sensor-clinical-mapping', 'figure'),
     Output('cohort-progression-analysis', 'figure'),
     Output('longitudinal-trends', 'figure')],
    [Input('patient-dropdown', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('analysis-mode', 'value')]
)
def update_all_visualizations(selected_patient, x_feature, y_feature, analysis_mode):
    
    if df_clean.empty:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return [empty_fig] * 6
    
    # Main correlation plot - Enhanced multi-modal scatter with clinical context
    fig_main = create_enhanced_correlation_plot(df_clean, x_feature, y_feature, selected_patient, analysis_mode)
    
    # Clinical-objective correlation - Direct clinical validation visualization
    fig_clinical = create_clinical_objective_correlation(df_clean, selected_patient)
    
    # Bilateral asymmetry analysis - Comprehensive bilateral motor assessment
    fig_bilateral = create_bilateral_asymmetry_analysis(df_clean, selected_patient)
    
    # Sensor-clinical mapping - Digital biomarker validation
    fig_sensor = create_sensor_clinical_mapping(df_clean, selected_patient)
    
    # Cohort progression analysis - Disease progression visualization
    fig_progression = create_cohort_progression_analysis(df_clean, selected_patient)
    
    # Longitudinal trends - Multi-modal temporal analysis
    fig_longitudinal = create_longitudinal_trends(df_clean, y_feature, selected_patient)
    
    return fig_main, fig_clinical, fig_bilateral, fig_sensor, fig_progression, fig_longitudinal

def create_enhanced_correlation_plot(df, x_feature, y_feature, selected_patient, analysis_mode):
    """Enhanced correlation plot with multi-modal insights - Comprehensive correlation visualization"""
    
    # Filter valid data
    valid_data = df.dropna(subset=[x_feature, y_feature])
    
    fig = px.scatter(
        valid_data,
        x=x_feature,
        y=y_feature,
        color='COHORT_NAME',
        size='CLINICAL_MOTOR_SEVERITY' if 'CLINICAL_MOTOR_SEVERITY' in valid_data.columns else None,
        hover_data=['PATNO', 'EVENT_ID', 'SEVERITY_CATEGORY', 'ENROLL_AGE'] if 'ENROLL_AGE' in valid_data.columns else ['PATNO', 'EVENT_ID'],
        title=f'Multi-Modal Analysis: {FEATURE_LABELS.get(y_feature, y_feature)} vs. {FEATURE_LABELS.get(x_feature, x_feature)}',
        color_discrete_map={
            "Parkinson's Disease": '#e74c3c',
            'Healthy Control': '#2ecc71', 
            'Prodromal': '#3498db',
            'SWEDD': '#f39c12',
            'Unknown': '#95a5a6'
        },
        marginal_x="violin" if analysis_mode == 'multimodal' else "histogram",
        marginal_y="violin" if analysis_mode == 'multimodal' else "histogram"
    )
    
    # Add statistical trend lines - Enhanced statistical analysis
    for cohort in valid_data['COHORT_NAME'].unique():
        cohort_data = valid_data[valid_data['COHORT_NAME'] == cohort]
        if len(cohort_data) > 2:
            # Linear regression
            correlation = np.corrcoef(cohort_data[x_feature], cohort_data[y_feature])[0,1]
            z = np.polyfit(cohort_data[x_feature], cohort_data[y_feature], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(cohort_data[x_feature].min(), cohort_data[x_feature].max(), 100)
            
            fig.add_trace(go.Scatter(
                x=x_trend, 
                y=p(x_trend),
                mode='lines',
                name=f'{cohort} trend (r={correlation:.2f})',
                line=dict(dash='dash', width=2),
                showlegend=True
            ))
    
    # Highlight selected patient - Enhanced patient trajectory visualization
    if selected_patient and selected_patient in valid_data['PATNO'].values:
        patient_data = valid_data[valid_data['PATNO'] == selected_patient]
        fig.add_trace(go.Scatter(
            x=patient_data[x_feature],
            y=patient_data[y_feature],
            mode='markers+lines',
            marker=dict(size=15, color='red', symbol='star', line=dict(width=2, color='black')),
            line=dict(color='red', width=3),
            name=f'Patient {selected_patient} trajectory'
        ))
    
    fig.update_layout(
        xaxis_title=FEATURE_LABELS.get(x_feature, x_feature),
        yaxis_title=FEATURE_LABELS.get(y_feature, y_feature),
        template='plotly_white'
    )
    
    return fig

def create_clinical_objective_correlation(df, selected_patient):
    """Clinical vs objective measurement correlation - Clinical validation visualization"""
    
    if 'CLINICAL_MOTOR_SEVERITY' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Clinical scores not available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    valid_data = df.dropna(subset=['CLINICAL_MOTOR_SEVERITY', 'OBJECTIVE_MOTOR_SCORE'])
    
    fig = px.scatter(
        valid_data,
        x='CLINICAL_MOTOR_SEVERITY',
        y='OBJECTIVE_MOTOR_SCORE',
        color='COHORT_NAME',
        title='Clinical Scores vs. Objective Motor Measurements',
        trendline="ols"
    )
    
    # Add correlation coefficient
    if len(valid_data) > 1:
        correlation = np.corrcoef(valid_data['CLINICAL_MOTOR_SEVERITY'], valid_data['OBJECTIVE_MOTOR_SCORE'])[0,1]
        fig.add_annotation(
            text=f"Overall Correlation: r = {correlation:.3f}",
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            bgcolor="white",
            bordercolor="gray"
        )
    
    fig.update_layout(
        xaxis_title='UPDRS-III Motor Score (Clinical)',
        yaxis_title='Objective Motor Impairment Score',
        template='plotly_white'
    )
    
    return fig

def create_bilateral_asymmetry_analysis(df, selected_patient):
    """Bilateral motor asymmetry analysis - Comprehensive asymmetry assessment"""
    
    valid_data = df.dropna(subset=['RA_AMP_U', 'LA_AMP_U'])
    
    fig = go.Figure()
    
    # Add scatter plot for each cohort
    for cohort in valid_data['COHORT_NAME'].unique():
        cohort_data = valid_data[valid_data['COHORT_NAME'] == cohort]
        fig.add_trace(go.Scatter(
            x=cohort_data['RA_AMP_U'],
            y=cohort_data['LA_AMP_U'],
            mode='markers',
            name=cohort,
            opacity=0.7
        ))
    
    # Add perfect symmetry line
    max_val = max(valid_data['RA_AMP_U'].max(), valid_data['LA_AMP_U'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Perfect Symmetry',
        line=dict(dash='dash', color='gray', width=2)
    ))
    
    # Add asymmetry zones - Clinical interpretation zones
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=max_val*0.8, y1=max_val,
        fillcolor="rgba(255,0,0,0.1)",
        line=dict(width=0),
    )
    
    fig.add_annotation(
        text="Right-dominant<br>asymmetry zone",
        x=max_val*0.6, y=max_val*0.9,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    fig.update_layout(
        title='Bilateral Arm Amplitude Analysis (Left vs Right)',
        xaxis_title='Right Arm Amplitude',
        yaxis_title='Left Arm Amplitude',
        template='plotly_white'
    )
    
    return fig

def create_sensor_clinical_mapping(df, selected_patient):
    """Digital sensor to clinical score mapping - Digital biomarker validation"""
    
    if 'SENSOR_MEAN' not in df.columns or 'CLINICAL_MOTOR_SEVERITY' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Sensor or clinical data not available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    valid_data = df.dropna(subset=['SENSOR_MEAN', 'CLINICAL_MOTOR_SEVERITY'])
    
    if len(valid_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No valid sensor-clinical pairs", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    fig = px.scatter(
        valid_data,
        x='SENSOR_MEAN',
        y='CLINICAL_MOTOR_SEVERITY',
        color='COHORT_NAME',
        title='Digital Sensor Response vs. Clinical Motor Severity',
        trendline="ols"
    )
    
    # Add correlation information - Enhanced statistical interpretation
    if len(valid_data) > 1:
        correlation = np.corrcoef(valid_data['SENSOR_MEAN'], valid_data['CLINICAL_MOTOR_SEVERITY'])[0,1]
        fig.add_annotation(
            text=f"Sensor-Clinical Correlation: r = {correlation:.3f}<br>{'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'} relationship",
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            bgcolor="white",
            bordercolor="gray"
        )
    
    fig.update_layout(
        xaxis_title='Digital Sensor Mean Response',
        yaxis_title='Clinical Motor Severity (UPDRS-III)',
        template='plotly_white'
    )
    
    return fig

def create_cohort_progression_analysis(df, selected_patient):
    """Cohort-based progression analysis - Disease progression visualization"""
    
    # Create severity distribution by cohort
    valid_data = df.dropna(subset=['COHORT_NAME', 'CLINICAL_MOTOR_SEVERITY'])
    
    if len(valid_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No clinical severity data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    fig = px.violin(
        valid_data,
        x='COHORT_NAME',
        y='CLINICAL_MOTOR_SEVERITY',
        color='COHORT_NAME',
        title='Motor Severity Distribution by Cohort',
        box=True,
        points="outliers"
    )
    
    # Add statistical annotations - Enhanced statistical comparison
    cohorts = valid_data['COHORT_NAME'].unique()
    if len(cohorts) >= 2:
        # Add mean lines for each cohort
        for cohort in cohorts:
            cohort_data = valid_data[valid_data['COHORT_NAME'] == cohort]
            mean_val = cohort_data['CLINICAL_MOTOR_SEVERITY'].mean()
            fig.add_hline(
                y=mean_val,
                line_dash="dash",
                annotation_text=f"{cohort}: μ={mean_val:.1f}",
                annotation_position="right"
            )
    
    fig.update_layout(
        xaxis_title='Patient Cohort',
        yaxis_title='Clinical Motor Severity (UPDRS-III)',
        template='plotly_white'
    )
    
    return fig

def create_longitudinal_trends(df, y_feature, selected_patient):
    """Enhanced longitudinal analysis - Multi-modal temporal analysis"""
    
    # Convert EVENT_ID to numeric for time ordering - Comprehensive visit mapping
    event_order = {
        'BL': 0, 'SC': 0, 'V01': 1, 'V02': 2, 'V04': 4, 'V06': 6, 'V08': 8, 
        'V09': 9, 'V10': 10, 'V12': 12, 'V14': 14, 'V15': 15, 'V17': 17, 'V18': 18, 'V19': 19
    }
    
    df_time = df.copy()
    df_time['EVENT_NUMERIC'] = df_time['EVENT_ID'].map(event_order)
    df_time = df_time.dropna(subset=['EVENT_NUMERIC', y_feature]).sort_values(['PATNO', 'EVENT_NUMERIC'])
    
    if len(df_time) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No longitudinal data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    fig = go.Figure()
    
    # Plot average trajectory for each cohort - Enhanced cohort trajectory visualization
    cohort_colors = {
        "Parkinson's Disease": '#e74c3c',
        'Healthy Control': '#2ecc71', 
        'Prodromal': '#3498db',
        'SWEDD': '#f39c12',
        'Unknown': '#95a5a6'
    }
    
    for cohort in df_time['COHORT_NAME'].unique():
        cohort_data = df_time[df_time['COHORT_NAME'] == cohort]
        if len(cohort_data) > 0:
            # Calculate mean and standard error by visit
            avg_by_visit = cohort_data.groupby('EVENT_NUMERIC')[y_feature].agg(['mean', 'std', 'count']).reset_index()
            avg_by_visit['se'] = avg_by_visit['std'] / np.sqrt(avg_by_visit['count'])
            
            color = cohort_colors.get(cohort, '#95a5a6')
            
            # Main trajectory line
            fig.add_trace(go.Scatter(
                x=avg_by_visit['EVENT_NUMERIC'],
                y=avg_by_visit['mean'],
                mode='lines+markers',
                name=f'{cohort} (n={cohort_data["PATNO"].nunique()})',
                line=dict(width=3, color=color),
                marker=dict(size=8)
            ))
            
            # Confidence intervals - Statistical confidence visualization
            fig.add_trace(go.Scatter(
                x=list(avg_by_visit['EVENT_NUMERIC']) + list(avg_by_visit['EVENT_NUMERIC'][::-1]),
                y=list(avg_by_visit['mean'] + avg_by_visit['se']) + list((avg_by_visit['mean'] - avg_by_visit['se'])[::-1]),
                fill='tonexty',
                mode='none',
                name=f'{cohort} ±SE',
                showlegend=False,
                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)'
            ))
    
    # Highlight selected patient trajectory - Enhanced patient-specific visualization
    if selected_patient and selected_patient in df_time['PATNO'].values:
        patient_time = df_time[df_time['PATNO'] == selected_patient]
        if len(patient_time) > 0:
            fig.add_trace(go.Scatter(
                x=patient_time['EVENT_NUMERIC'],
                y=patient_time[y_feature],
                mode='lines+markers',
                name=f'Patient {selected_patient} (Individual)',
                line=dict(color='red', width=4, dash='dot'),
                marker=dict(size=12, color='red', symbol='diamond')
            ))
    
    # Add clinical interpretation zones - Clinical context visualization
    if y_feature == 'CLINICAL_MOTOR_SEVERITY':
        fig.add_hrect(y0=0, y1=20, fillcolor="rgba(46,204,113,0.1)", annotation_text="Mild", annotation_position="left")
        fig.add_hrect(y0=20, y1=40, fillcolor="rgba(241,196,15,0.1)", annotation_text="Moderate", annotation_position="left")
        fig.add_hrect(y0=40, y1=100, fillcolor="rgba(231,76,60,0.1)", annotation_text="Severe", annotation_position="left")
    
    fig.update_layout(
        title=f'Longitudinal {FEATURE_LABELS.get(y_feature, y_feature)} Trends by Cohort',
        xaxis_title='Visit Number',
        yaxis_title=FEATURE_LABELS.get(y_feature, y_feature),
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Add visit labels - Enhanced temporal context
    visit_labels = {0: 'Baseline', 4: '6 Months', 8: '12 Months', 12: '18 Months', 15: '2 Years'}
    for visit_num, label in visit_labels.items():
        if visit_num in df_time['EVENT_NUMERIC'].values:
            fig.add_vline(x=visit_num, line_dash="dot", line_color="gray", 
                         annotation_text=label, annotation_position="top")
    
    return fig

# Run the App
if __name__ == '__main__':
    print("Starting Enhanced Multi-Modal Parkinson's Dashboard...")
    print("Features: Clinical scores, gait data, digital sensors, demographics")
    print("Analysis modes: Cross-sectional, longitudinal, multi-modal")
    app.run(debug=True)