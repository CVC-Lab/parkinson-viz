# Multi-Modal Parkinson's Disease Motion Visualization System

A real-time interactive platform for visualizing Parkinson's disease motor symptoms through an articulated **3D motion model** and multi-modal data analysis.


## Live Demo

**Web Application**: [https://cvc-lab.github.io/parkinson-viz/](https://cvc-lab.github.io/parkinson-viz/)

The live app is a **static, client-side** build (no backend): a Three.js 3D figure plus
Plotly.js charts, hosted on GitHub Pages from the `github-pages` branch. The original
Python/**Dash** version (`app.py`) is kept as a server-rendered alternative, and its data
pipeline also produces the JSON the static app consumes (via `convert_data_to_json.py`).

## System Overview

This system transforms clinical Parkinson's disease data into intuitive, real-time anatomical motion visualizations. It integrates multiple data modalities to provide comprehensive patient assessment through:

- **Real-time motion silhouettes** with anatomically-accurate human figures
- **Multi-modal data integration** (gait, clinical scores, digital sensors)
- **Interactive analysis tools** for bilateral movement comparison
- **Clinical feature engineering** with evidence-based composite indices
- **60fps smooth animation** (local deployment)

### Key Features

**Articulated 3D Model**: Forward-kinematics human figure (orbit / zoom / camera presets) whose joints move with the patient's data  
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

- **Live frontend (static)**: Three.js for the 3D motion model + Plotly.js for the 2D charts, vanilla ES modules — no backend
- **3D animation**: forward-kinematics joint hierarchy driven by `requestAnimationFrame`
- **Data pipeline**: Python (Pandas, NumPy, SciPy) in `app.py`, exported to JSON via `convert_data_to_json.py`
- **Original app**: Dash (Python) + Plotly server-rendered version in `app.py`
- **Deployment**: GitHub Pages (static, `github-pages` branch); the Dash app is separately deployable to a server

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
The PPMI study spans PD, Healthy Control, Prodromal, and SWEDD cohorts. The data
**currently shipped in this app** contains only **Parkinson's Disease (49)** and
**Prodromal (37)** participants.

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

### 3D Motion Model

The figure (`js/figure3d.js`) is a forward-kinematics joint hierarchy
(pelvis → spine → neck → head, shoulders → elbows → wrists, hips → knees → ankles → feet)
built from Three.js capsule segments. Because every limb is a child group of its proximal
joint, rotating a joint moves the whole chain and the body can never come apart — motion is
joint *rotation*, not independent translation of polygons.

### Real-Time Animation

A single `requestAnimationFrame` loop advances a motion clock; `js/motion.js` maps the
selected participant's data to a pose (a set of joint angles) each frame, and the figure
applies it. The same clock drives the phase marker on the gait-cycle chart. OrbitControls
provide drag-to-orbit / scroll-to-zoom plus Front / Side / ¾ / Top camera presets.

### Motion From Data

Motion is grounded in real PPMI columns (`js/motion.js`). For gait, arms swing opposite
their same-side leg with amplitude from each arm's measured swing, cadence sets stride
frequency, gait speed sets stride length, UPDRS-III / Hoehn-Yahr severity adds stoop and
shuffle, and the tremor sub-scores add a fine hand tremor:

```javascript
const shoR = -armAmp.r * Math.sin(th) * (1 - 0.15 * severity);  // right arm
const shoL =  armAmp.l * Math.sin(th) * (1 - 0.15 * severity);  // left arm (asymmetry visible)
```

Balance uses cohort-normalized postural sway; TUG runs a sit → stand → walk → turn → sit cycle.

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

### Research & Education (not for clinical use)
This is a research and educational visualization, **not a clinical decision-support
tool or a medical device** — it is not validated for diagnosis, treatment planning, or
patient care.
- **Objective patterns**: visualize clinical scores and movement metrics
- **Education**: illustrate Parkinsonian movement patterns
- **Research**: explore cohort patterns and biomarkers

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
