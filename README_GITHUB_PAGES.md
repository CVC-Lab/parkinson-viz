# Parkinson's Disease Motion Visualization - GitHub Pages Version

## Overview

This is the **client-side JavaScript version** of the Multi-Modal Parkinson's Disease Motion Visualization system. This version runs entirely in the browser and can be hosted on GitHub Pages for free!

### Key Features

- ✅ **100% Client-Side** - No server required
- ✅ **Smooth 60fps Animation** - Better performance than cloud deployment
- ✅ **Free Hosting** - GitHub Pages deployment
- ✅ **Offline Capable** - Works once data is loaded
- ✅ **Real-time Motion Silhouettes** - Anatomically accurate human figures
- ✅ **Multi-Modal Data Integration** - Gait, clinical scores, digital sensors
- ✅ **Interactive Analysis** - Patient selection, feature correlation

## Quick Start

### Option 1: Deploy to GitHub Pages

1. **Push this branch to GitHub:**
   ```bash
   git add .
   git commit -m "Add GitHub Pages version"
   git push origin github-pages
   ```

2. **Enable GitHub Pages:**
   - Go to your repository on GitHub
   - Click **Settings** → **Pages**
   - Under "Source", select branch: `github-pages`
   - Click **Save**
   - Your site will be live at: `https://YOUR_USERNAME.github.io/parkinson-viz/`

### Option 2: Run Locally

1. **Start a local server** (required for ES6 modules):
   ```bash
   # Python 3
   python3 -m http.server 8000

   # OR Python 2
   python -m SimpleHTTPServer 8000

   # OR Node.js
   npx http-server -p 8000
   ```

2. **Open in browser:**
   ```
   http://localhost:8000
   ```

## Project Structure

```
parkinson-viz/
├── index.html              # Main HTML page
├── css/
│   └── styles.css         # All styles
├── js/
│   ├── app.js            # Main application logic
│   ├── data-loader.js    # Data loading and processing
│   └── motion-generator.js # Motion silhouette generator
├── data/
│   └── merged_data.json  # Pre-processed patient data (0.48 MB)
├── .nojekyll             # Prevents Jekyll processing
└── README_GITHUB_PAGES.md # This file
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Visualization | [Plotly.js](https://plotly.com/javascript/) |
| Animation | `requestAnimationFrame()` (60fps) |
| Data Format | JSON (pre-processed from CSV) |
| Modules | ES6 Modules |
| Hosting | GitHub Pages (static) |

## Data Conversion

The Python data processing pipeline is preserved and runs during build:

```bash
# Re-generate JSON data from CSV sources
/Users/ryanfarell/miniconda3/envs/dev/bin/python convert_data_to_json.py
```

This script:
- Loads all CSV datasets (gait, UPDRS, demographics, sensors)
- Merges on patient ID and visit
- Performs feature engineering
- Exports to `data/merged_data.json`

## Features

### 1. Real-Time Motion Silhouettes
- Anatomically accurate 8-head proportional human figures
- 16 body parts with independent motion
- Motion types: Gait, TUG Test, Balance, Free Motion
- Patient-specific motion parameters

### 2. Interactive Controls
- **Patient Selection** - 86 patients from PPMI dataset
- **Motion Test Type** - Gait, TUG, Balance, Free
- **Animation Speed** - 0.1x to 3.0x
- **Play/Pause/Reset** - Full animation control
- **Axis Selection** - Any feature vs. any feature

### 3. Multi-Modal Visualizations
- **Correlation Analysis** - Main scatter plot with cohort grouping
- **Bilateral Asymmetry** - Left vs. right arm comparison
- **Gait Cycle** - Phase-locked arm swing patterns
- **Motion Quality** - Radar chart assessment

### 4. Clinical Data Integration
- **Gait Measurements** - Speed, asymmetry, arm amplitude
- **UPDRS Scores** - Motor exam and patient-reported
- **Digital Sensors** - Drawing, voice, tapping tests
- **Demographics** - Age, cohort, clinical staging

## Performance

| Metric | GitHub Pages | Original Cloud |
|--------|--------------|----------------|
| Animation FPS | 60fps | ~10fps |
| Data Processing | Client-side | Server-side |
| Latency | Zero | High |
| Cost | Free | Cloud fees |

**Why is this faster?**
- All processing happens locally in the browser
- No network round-trips for animation frames
- Native `requestAnimationFrame()` optimization

## Browser Compatibility

- ✅ Chrome/Edge 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ⚠️ Requires ES6 module support
- ⚠️ Requires JavaScript enabled

## Customization

### Add New Features

Edit `js/data-loader.js`:
```javascript
export const FEATURE_LABELS = {
    'YOUR_FEATURE': 'Your Feature Label',
    // ...
};
```

### Modify Motion Parameters

Edit `js/motion-generator.js`:
```javascript
calculateGaitMotion(leftArmAmp, rightArmAmp, ...) {
    // Adjust motion calculations
}
```

### Update Styling

Edit `css/styles.css`:
```css
.control-panel {
    /* Customize colors, layout, etc. */
}
```

## Data Privacy

- ✅ All data is de-identified (PATNO IDs only)
- ✅ Data from public PPMI dataset
- ✅ No PHI or sensitive information
- ✅ Client-side processing (no data leaves browser)

## Updating Data

When you have new CSV files:

1. Replace files in respective directories:
   - `Motor_Assessments/`
   - `Subject_Characteristics/`
   - `Digital_Sensor/`

2. Run conversion:
   ```bash
   /Users/ryanfarell/miniconda3/envs/dev/bin/python convert_data_to_json.py
   ```

3. Commit and push:
   ```bash
   git add data/merged_data.json
   git commit -m "Update patient data"
   git push
   ```

## Troubleshooting

### "CORS Error" when running locally
- **Cause**: ES6 modules require a web server
- **Fix**: Use `python3 -m http.server` or similar

### Visualizations not updating
- **Check**: Browser console for errors
- **Fix**: Clear cache and reload (Cmd+Shift+R / Ctrl+Shift+R)

### Animation is choppy
- **Cause**: Too many browser tabs or low-end device
- **Fix**: Close other tabs, reduce animation speed

### Data not loading
- **Check**: `data/merged_data.json` exists and is valid JSON
- **Fix**: Re-run `convert_data_to_json.py`

## Development

### Making Changes

1. Edit files in `js/`, `css/`, or `index.html`
2. Test locally with a web server
3. Commit and push to `github-pages` branch
4. Changes appear on GitHub Pages in ~1 minute

### Adding New Visualizations

1. Add new plot container in `index.html`
2. Create plot function in `js/app.js`
3. Call from `updateAllVisualizations()`

Example:
```javascript
function updateMyNewPlot() {
    const traces = [/* ... */];
    const layout = {/* ... */};
    Plotly.newPlot('my-new-plot', traces, layout);
}
```

## Credits

- **Data Source**: PPMI (Parkinson's Progression Markers Initiative)
- **Visualization**: Plotly.js
- **Animation**: JavaScript requestAnimationFrame
- **Original Python Version**: Dash + Plotly Python

## License

This project uses publicly available PPMI data. Please cite PPMI appropriately if used for research.

## Support

For issues or questions:
1. Check browser console for errors
2. Verify data files exist
3. Test in latest Chrome/Firefox
4. Open GitHub issue with details

---

**Live Demo**: https://YOUR_USERNAME.github.io/parkinson-viz/
