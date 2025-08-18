# Multi-Modal Parkinson's Disease Motion Visualization System

A real-time interactive platform for visualizing Parkinson's disease motor symptoms through anatomically-accurate motion silhouettes and multi-modal data analysis.


## Live Demo

**Web Application**: [https://parkinson-viz.onrender.com/](https://parkinson-viz.onrender.com/)

**Performance Note**: The cloud deployment has animation limitations due to browser ↔ server latency. For optimal 60fps real-time motion visualization, **local deployment is recommended**.

## System Overview

This system transforms clinical Parkinson's disease data into intuitive, real-time anatomical motion visualizations. It integrates multiple data modalities to provide comprehensive patient assessment through:

- **Real-time motion silhouettes** with anatomically-accurate human figures
- **Multi-modal data integration** (gait, clinical scores, digital sensors)
- **Interactive analysis tools** for bilateral movement comparison
- **Clinical feature engineering** with evidence-based composite indices
- **60fps smooth animation** (local deployment)

### Key Features

**Anatomically-Accurate Silhouettes**: 8-head proportional human figures with proper joint articulation  
**Patient-Specific Animation**: Motion patterns derived from individual gait measurements  
**Multi-Modal Analysis**: Gait data + UPDRS scores + Digital sensors + Demographics  
**Real-Time Controls**: Play/Pause/Speed controls with phase-locked motion  
**Clinical Insights**: Bilateral asymmetry, movement quality, severity staging  
**Interactive Visualizations**: Correlation plots, radar charts, motion cycles  

## Architecture

### End-to-End Pipeline

```
Raw Data Sources → Data Integration → Feature Engineering → Real-Time Visualization
     ↓                    ↓                 ↓                      ↓
- Gait Data         → Multi-Dataset    → Clinical Scores    → Motion Silhouettes
- UPDRS Scores      → Merge & Clean    → Motion Features    → Interactive Plots  
- Digital Sensors   → Quality Control  → Composite Indices  → Animation Controls
- Demographics      → Missing Data     → Bilateral Analysis → Performance Metrics
```

### Technical Components

- **Frontend**: Dash (Python) + Plotly.js for interactive visualization
- **Backend**: Python with Pandas, NumPy, SciPy for data processing
- **Animation**: Custom SVG-based silhouette generator with real-time updates
- **Data Processing**: Multi-threaded callback system for smooth animations
- **Deployment**: Render.com cloud platform + local deployment support

## Data Sources

### Parkinson's Disease Cohort

The system uses comprehensive multi-modal datasets from Parkinson's disease research:

#### **Primary Datasets**
- **Motor Assessments** (192 patients): Gait analysis with 60+ motion parameters
  - Arm swing amplitudes (LA_AMP_U, RA_AMP_U)
  - Gait speed and cadence (SP_U, CAD_U)
  - Movement asymmetry (ASA_U) and smoothness (JERK)
  - Timed Up & Go (TUG) test measurements

- **UPDRS Clinical Scores**: Standardized severity assessments
  - Part II: Patient-reported motor symptoms
  - Part III: Clinician-assessed motor examination
  - Part IV: Motor complications

- **Digital Sensor Data** (108,901 measurements): High-frequency behavioral signals
  - Drawing tests (spirals, lines, circles)
  - Voice analysis and finger tapping
  - Daily activity monitoring

- **Demographics & Status**: Patient stratification variables
  - Age, sex, handedness, race/ethnicity
  - Cohort definitions and enrollment status

#### **Patient Populations**
- **PD Patients**: Primary Parkinson's disease cohort
- **Healthy Controls**: Neurologically normal comparison group
- **Prodromal**: Early-stage/at-risk individuals
- **SWEDD**: Subjects Without Evidence of Dopaminergic Deficit

## Local Installation & Setup

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Required packages
pip install dash plotly pandas numpy scipy
```

### Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/erickim73/parkinson-viz.git
cd parkinson-viz
```

2. **Create Virtual Environment** (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare Data Directory Structure**
```
parkinson-motion-viz/
├── Curated_Data_Cuts/
|   ├── PPMI_Curated_Data_Cut_Public_20241211.xlsx
├──Medical_History
|   ├──Features_of_Parkinsonism_06_Jan2025.csv
|   ├──Neuological_Exam_05Jan2025.csv
|   ├──Other_Clinical_Features_06Jan2025.csv
├── Motor_Assessments/
│   ├── Gait_Data___Arm_swing_06Jan2025.csv
|   ├──Gait_Substudy_Gait_Mobility_Assessment_and_Measurement_06Jan2025.csv
│   ├── MDS_UPDRS_Part_II__Patient_Questionnaire_06Jan2025.csv
│   ├── MDS-UPDRS_Part_III_06Jan2025.csv
│   └── MDS-UPDRS_Part_IV__Motor_Complications_06Jan2025.csv
├── Study_Docs/
|   ├──Code_List_-__Annotated__08Jan2025.csv
|   ├──Data_Dictionary_-__Annotated__08Jan2025.csv
├── Subject_Characteristics/
│   ├── Demographics_08Jan2025.csv
│   └── Participant_Status_08Jan2025.csv
└── app.py
```

5. **Run the Application**
```bash
python app.py
```

6. **Access the Application**
Open your browser and navigate to: `http://127.0.0.1:8050/`

### Expected Output
```
Loading datasets...
✓ Loaded gait data: (192, 60)
✓ Loaded UPDRS III: (xxx, xx)
✓ Loaded UPDRS II: (xxx, xx)
✓ Loaded demographics: (xxx, xx)
✓ Loaded status: (xxx, xx)
✓ Final merged dataset: (xxx, xx)
✓ Available patients: xxx
Starting Enhanced Multi-Modal Parkinson's Dashboard...
Dash is running on http://127.0.0.1:8050/
```

## How to Use the System

### 1. **Patient Selection**
- Choose a patient from the dropdown menu
- The system loads patient-specific gait and clinical data
- Motion silhouette updates to reflect individual movement patterns

### 2. **Motion Test Types**
- **Gait/Walking Test**: Continuous walking motion with arm swing
- **TUG Test**: Timed Up & Go sequence (sit→stand→walk→turn→sit)
- **Postural Sway/Balance**: Standing balance with center-of-mass movement
- **Free Motion**: Unconstrained movement patterns

### 3. **Animation Controls**
- **Play/Pause**: Control real-time motion animation
- **Speed**: Adjust animation speed (0.1x to 3.0x)
- **Reset**: Return to initial animation phase

### 4. **Analysis Features**
- **Correlation Analysis**: X/Y axis feature selection for scatter plots
- **Bilateral Comparison**: Left vs. right arm movement analysis
- **Motion Quality**: Radar chart of movement characteristics
- **Gait Cycle**: Phase-locked motion patterns over time

## 🔧 System Architecture Deep Dive

### Motion Silhouette Generation

The core innovation is the anatomically-accurate motion silhouette system:

```python
class MotionSilhouetteGenerator:
    """Anatomically-correct motion visualization"""
    
    def create_anatomical_silhouette(self):
        # 8-head figure proportions with realistic body segments
        return {
            'head': {...},           # Proportional head structure  
            'torso': {...},          # Central body reference
            'left_upper_arm': {...}, # Bilateral arm segments
            'right_upper_arm': {...},
            'left_forearm': {...},
            'right_forearm': {...},
            'left_hand': {...},
            'right_hand': {...},
            'left_thigh': {...},     # Leg segments with joints
            'right_thigh': {...},
            # ... additional body parts
        }
```

### Real-Time Animation System

The system uses a sophisticated 4-callback architecture for smooth 60fps animation:

1. **Patient Data Store** (Low Frequency): Caches patient data to avoid expensive lookups
2. **Animation Control** (User Triggered): Manages play/pause/speed controls
3. **Time Phase Ticker** (High Frequency - 100ms): Lightweight phase incrementation
4. **Silhouette Renderer** (Data Driven): Main rendering engine for motion visualization

### Motion Pattern Algorithms

Patient-specific motion is calculated using clinical measurements:

```python
def calculate_gait_motion(left_arm_amp, right_arm_amp, gait_speed, time_phase, asymmetry):
    # Normalize amplitudes to realistic range
    left_swing = (left_arm_amp / 50.0) * math.sin(time_phase) * 0.5
    right_swing = (right_arm_amp / 50.0) * math.sin(time_phase + math.pi) * 0.5
    
    # Apply clinical asymmetry
    asymmetry_factor = min(asymmetry / 10.0, 0.3)
    left_swing *= (1 + asymmetry_factor)
    right_swing *= (1 - asymmetry_factor)
    
    # Synchronize leg motion
    speed_factor = min(gait_speed, 1.5)
    leg_phase = time_phase * speed_factor
    left_leg_swing = math.sin(leg_phase + math.pi) * 0.3
    right_leg_swing = math.sin(leg_phase) * 0.3
```

## Performance Considerations

### Local vs Cloud Deployment

| Aspect | Local Deployment | Cloud (Render.com) |
|--------|------------------|-------------------|
| **Animation FPS** | 60fps (smooth) | ~10fps (laggy) |
| **Data Processing** | Direct memory access | Browser ↔ Server latency |
| **User Experience** | Optimal | Limited |
| **Setup Required** | Python environment | None |
| **Cost** | Free | Free tier (with limitations) |

### Why Local is Recommended

The cloud deployment suffers from a fundamental architectural challenge:
- **Issue**: Real-time animation data travels browser → server → browser on each frame
- **Impact**: Noticeable lag, reduced frame rate, poor user experience
- **Solution**: Local deployment eliminates network latency entirely

### Optimization Strategies

```python
# Efficient data handling for large datasets
def enhance_merged_data(self, merged):
    """Optimized feature engineering"""
    # Select only essential columns to minimize memory usage
    essential_cols = ['PATNO', 'ASA_U', 'SP_U', 'RA_AMP_U', 'LA_AMP_U']
    
    # Vectorized operations for speed
    merged['ARM_ASYMMETRY'] = abs(merged['RA_AMP_U'] - merged['LA_AMP_U']) / \
                              (merged['RA_AMP_U'] + merged['LA_AMP_U'] + 1e-6)
```


## Clinical Applications

### Research Applications
- **Movement Biomarker Discovery**: Novel patterns in motion data
- **Treatment Response Monitoring**: Visualize therapy effectiveness
- **Disease Progression Tracking**: Longitudinal movement changes
- **Cohort Comparison Studies**: Control vs. patient populations

### Clinical Decision Support
- **Objective Assessment**: Transform clinical scores into visual patterns
- **Patient Education**: Help patients understand their movement patterns
- **Therapy Planning**: Identify specific movement impairments
- **Progress Monitoring**: Track improvement over time

### Educational Use
- **Medical Training**: Visualize Parkinson's motor symptoms
- **Patient Understanding**: Intuitive representation of clinical data
- **Research Presentation**: Compelling visualization of findings

## Future Enhancements

### Technical Roadmap

**Immediate Improvements:**
- [ ] Client-side animation rendering for cloud deployment
- [ ] Progressive data loading for faster initialization
- [ ] Enhanced mobile responsiveness
- [ ] Export functionality for motion videos

**Advanced Features:**
- [ ] 3D motion visualization with depth perception
- [ ] Machine learning severity prediction from motion patterns
- [ ] Multi-patient comparison views
- [ ] Real-time sensor data integration

**Clinical Integration:**
- [ ] DICOM compatibility for medical imaging integration
- [ ] HL7 FHIR standards for clinical data exchange
- [ ] Automated report generation
- [ ] Clinical decision support recommendations
