# FSAE Telemetry Analysis Tool

A comprehensive Python application for analyzing FSAE car telemetry data with support for aero vs no-aero configuration comparison.

## Features

- **Automatic Best Lap Extraction**: Uses beacon markers and segment times to identify the best lap from session data
- **Aero Configuration Comparison**: Side-by-side comparison of aerodynamic vs non-aerodynamic setups
- **Individual Analysis**: Detailed analysis of single telemetry files
- **Multiple Visualizations**:
  - Shock travel and velocity percent histograms (FL, FR, RL, RR)
  - GPS speed, yaw rate, and ECU RPM time series
  - GPS speed, yaw rate, and ECU RPM percent histograms
  - GG diagram (lateral vs longitudinal acceleration)
  - GPS speed vs yaw rate scatter plot
  - GPS lateral acceleration vs yaw rate scatter plot
  - GPS speed vs wheel speed (RL and RR) correlation
  - Best-lap GPS track map overlay

## Installation

### Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib

### Setup

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Place your CSV files in the same directory as the script, then:

```bash
python3 telemetry_analysis.py
```

The application will prompt you to choose an analysis mode:
1. **Comparison Mode** (default): Analyzes aero vs no-aero configurations side-by-side
2. **Individual Mode**: Analyzes each file separately

For the included sample files, you can run the dashboard app directly:

```bash
python3 app.py
```

By default, this creates `dashboard_comparison.html` and compares `265.csv` as no-aero against both aero files, `266.csv` and `273.csv`. You can also pass file groups explicitly:

```bash
python3 app.py --no-aero 265.csv --aero 266.csv 273.csv
```

Use individual mode to create one dashboard section per run:

```bash
python3 app.py --mode individual
```

List available graph IDs:

```bash
python3 app.py --list-graphs
```

Generate only selected graphs:

```bash
python3 app.py --mode comparison --graphs track_map gg speed_yaw
```

### Custom Dashboard Graphs

Graph names, order, and available graph IDs live in `dashboard_config.py`.

To rename a graph, edit its `name`.
To remove a graph from the default dashboard, remove or comment out its entry.
To add a graph, add a new entry with a unique `id`, `name`, and renderer method names, then implement those methods in `telemetry_analysis.py`.

### Interactive Upload App

Run the browser-based app:

```bash
streamlit run streamlit_app.py
```

In the app you can:
- Upload CSV files
- Assign each uploaded file as `Aero` or `No Aero`
- Switch between `comparison` and `individual` modes
- Select which graphs to generate
- Change the all-laps threshold percentage
- View plots directly in the browser

### CSV File Format

The script accepts AiM telemetry CSV exports with the following expected columns:
- **Time**: Sample timestamp (seconds)
- **GPS Speed**: Vehicle speed from GPS (km/h)
- **GPS LatAcc**: Lateral acceleration (g)
- **GPS LonAcc**: Longitudinal acceleration (g)
- **YawRate**: Vehicle yaw rate (deg/s)
- **ECU RPM**: Engine RPM
- **ECU WH SP RL**: Right-left wheel speed (km/h)
- **ECU WH SP RR**: Right-rear wheel speed (km/h)
- **Shock positions and velocities**: FL/FR/RL/RR shock travel and velocity

### Output Files

The script generates two PNG plot sets in comparison mode:
- `*_best_lap_comparison.png`: fastest valid complete no-aero lap vs fastest valid complete aero lap
- `*_all_laps_120pct_comparison.png`: every valid lap within 120% of each file's best lap
- `dashboard_comparison.html`: browsable comparison dashboard
- `dashboard_individual.html`: browsable individual-runs dashboard when using `--mode individual`

1. **shock_analysis_*.png**: Percent histogram grids showing shock travel and velocity distributions for all four corners
2. **speed_yaw_rpm_*.png**: Time-series plots of speed, yaw rate, and RPM
3. **speed_yaw_rpm_histograms_*.png**: Percent distribution histograms for speed, yaw rate, and RPM
4. **gg_diagram_*.png**: Lateral vs longitudinal acceleration scatter plot (competitive envelope visualization)
5. **speed_vs_yaw_*.png**: Speed vs yaw rate correlation
6. **latacc_vs_yaw_*.png**: Lateral acceleration vs yaw rate correlation
7. **speed_vs_wheelspeed_*.png**: GPS speed vs individual wheel speeds (RL and RR)
8. **track_map_*.png**: GPS track map overlay

## Example Output

For a session with both configurations:
```
Loading 265...
  Loaded 11800 samples
Loading 266...
  Loaded 3340 samples
Loading 273...
  Loaded 1240 samples

Extracting best lap from 265...
  Ignored 3 incomplete/outlier segment(s)
  Best lap time: 26.256s (segment 17)
  Best lap distance: 407.7m
  Extracted 525 samples for best lap

Extracting best lap from 266...
  Ignored 2 incomplete/outlier segment(s)
  Best lap time: 26.082s (segment 4)
  Best lap distance: 409.9m
  Extracted 522 samples for best lap

Extracting best lap from 273...
  Ignored 2 incomplete/outlier segment(s)
  Best lap time: 25.533s (segment 2)
  Best lap distance: 406.5m
  Extracted 511 samples for best lap

Comparing best laps: No Aero (265) vs Aero (266+273)
  [8 best-lap plots generated with comparison overlays]

Comparing all laps within 120% of each file's best lap: No Aero (265) vs Aero (266+273)
  [8 all-laps plots generated with comparison overlays]
```

## How Best Lap Detection Works

The tool uses the **Beacon Markers** and **Segment Times** metadata embedded in AiM CSV files:

1. Reads segment times from the CSV metadata (e.g., "0:28.538", "0:25.533", "0:07.928")
2. Measures each segment's GPS distance and sample count
3. Ignores incomplete/outlier segments, including out-laps, short partial laps at the end of a run, and interrupted laps
4. Identifies the shortest complete segment as the best lap
5. Uses beacon markers (time points) to extract only that lap's data and reset its time axis to start at 0 seconds

## Understanding the Plots

### Shock Analysis
- **Purpose**: Identify suspension behavior differences
- **Y-axis**: Frequency of occurrence
- **X-axis**: Shock position (mm) or velocity (mm/s)
- **Insight**: Aero setups may show different shock usage patterns due to increased downforce

### GG Diagram
- **Purpose**: Visualize vehicle performance envelope
- **X-axis**: Lateral acceleration (g)
- **Y-axis**: Longitudinal acceleration (g)
- **Reference circles**: 0.5g, 1.0g, 1.5g, 2.0g limits
- **Insight**: Aero enables higher acceleration limits, especially in cornering

### Speed vs Yaw Rate
- **Purpose**: Understand yaw response characteristics
- **Insight**: Higher speeds with lower yaw rates = more stable, efficient cornering

### Speed vs Wheel Speed
- **Purpose**: Detect slip behavior
- **Insight**: Gaps between GPS speed and wheel speed indicate slip events

## Advanced Usage

### Programmatic Access

```python
from telemetry_analysis import FSAETelemetryAnalyzer

analyzer = FSAETelemetryAnalyzer('/path/to/data')
analyzer.analyze(mode='comparison')
```

### Custom Analysis

Modify the visualization methods in `telemetry_analysis.py` to add custom plots or adjust figure sizes, colors, and metrics.

## Troubleshooting

**"No files found"**: Ensure CSV files are in the same directory as the script

**"Could not find header row"**: Verify the CSV file is in AiM format with the expected header structure

**Missing columns in plots**: Some telemetry data may not include all sensor columns. The script gracefully skips unavailable data.

## License

Use freely for FSAE analysis and development.
