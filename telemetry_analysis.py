"""
FSAE Telemetry Analysis Tool
Analyzes shock data, acceleration, and vehicle dynamics
Supports aero vs no-aero comparison and individual analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import warnings
import os
from pathlib import Path
from html import escape

from dashboard_config import CHANNEL_ALIASES, DASHBOARD_GRAPHS, DASHBOARD_TITLE
from graph_renderers import GraphRendererFactory

warnings.filterwarnings('ignore')

MM_PER_S2_PER_G = 9806.65

class FSAETelemetryAnalyzer:
    def __init__(
        self,
        data_dir,
        files=None,
        graph_ids=None,
        config_labels=None,
        lap_threshold=1.2,
        extra_graphs=None,
        comparison_lap_sets=None,
    ):
        """Initialize the analyzer with data directory"""
        self.data_dir = Path(data_dir)
        self.files = [Path(file) for file in files] if files else list(self.data_dir.glob('*.csv'))
        self.graph_ids = graph_ids
        self.config_labels = config_labels or {}
        self.lap_threshold = lap_threshold
        self.extra_graphs = extra_graphs or []
        self.comparison_lap_sets = comparison_lap_sets or ['best', 'all']
        self.data = {}
        self.best_laps = {}
        self.generated_plots = []
        
    def load_csv(self, filepath):
        """Load and parse AiM CSV file, skipping metadata rows"""
        import csv
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
        
        # Extract metadata first
        metadata = {}
        beacon_markers = []
        segment_times = []
        
        for row in lines[:20]:  # Metadata is in first ~20 rows
            if len(row) >= 2:
                key = row[0].strip('"').strip()
                if key == 'Beacon Markers':
                    beacon_markers = [float(x.strip('"').strip()) for x in row[1:] if x.strip('"').strip()]
                elif key == 'Segment Times':
                    # Convert "0:25.533" format to seconds
                    for val in row[1:]:
                        val = val.strip('"').strip()
                        if val:
                            try:
                                parts = val.split(':')
                                if len(parts) == 3:
                                    seg_time = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                                elif len(parts) == 2:
                                    seg_time = int(parts[0]) * 60 + float(parts[1])
                                else:
                                    seg_time = float(parts[0])
                                segment_times.append(seg_time)
                            except:
                                pass
                else:
                    metadata[key] = row[1].strip('"').strip() if len(row) > 1 else ''
        
        # Find header row (first row with "Time" as first column and many fields)
        header_idx = None
        for i, row in enumerate(lines):
            if len(row) > 10 and row[0].strip('"').strip() == 'Time':
                header_idx = i
                break
        
        if header_idx is None:
            raise ValueError("Could not find header row in CSV")
        
        # Extract headers and remove quotes
        headers = [col.strip('"').strip() for col in lines[header_idx]]
        
        # Skip units row (idx + 1) and empty rows, then collect data
        data_start_idx = header_idx + 2
        
        # Collect all data rows
        data_rows = []
        for row in lines[data_start_idx:]:
            if len(row) == 0 or (len(row) == 1 and row[0].strip() == ''):
                continue
            if len(row) == len(headers):
                # Clean the values
                cleaned_row = [val.strip('"').strip() for val in row]
                data_rows.append(cleaned_row)
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Convert numeric columns
        for col in df.columns:
            if col != 'Gear Position':  # Skip categorical column
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        df = df.reset_index(drop=True)
        
        # Store metadata for later use
        df.attrs['beacon_markers'] = beacon_markers
        df.attrs['segment_times'] = segment_times
        df.attrs['metadata'] = metadata
        
        return df
    
    def load_all_files(self):
        """Load all CSV files from directory"""
        for filepath in sorted(self.files):
            try:
                filepath = filepath if filepath.is_absolute() else self.data_dir / filepath
                label = filepath.stem
                print(f"Loading {label}...")
                self.data[label] = self.load_csv(filepath)
                print(f"  Loaded {len(self.data[label])} samples")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

    def _normalize_channel_name(self, name):
        """Normalize a channel/column name for fuzzy matching."""
        return ''.join(char.lower() for char in str(name) if char.isalnum())

    def resolve_channel(self, df, channel_id, required=False):
        """
        Resolve a stable channel id or raw column name to the actual DataFrame column.

        Add alternate raw names in dashboard_config.CHANNEL_ALIASES when an export
        uses different names for the same sensor.
        """
        if channel_id in df.columns:
            return channel_id

        candidates = CHANNEL_ALIASES.get(channel_id, [channel_id])
        for candidate in candidates:
            if candidate in df.columns:
                return candidate

        normalized_columns = {self._normalize_channel_name(column): column for column in df.columns}
        for candidate in candidates:
            normalized = self._normalize_channel_name(candidate)
            if normalized in normalized_columns:
                return normalized_columns[normalized]

        normalized_channel = self._normalize_channel_name(channel_id)
        for normalized_column, column in normalized_columns.items():
            if normalized_channel and (normalized_channel in normalized_column or normalized_column in normalized_channel):
                return column

        if required:
            raise KeyError(f"Could not resolve channel '{channel_id}'. Available columns: {', '.join(df.columns)}")
        return None

    def channel_series(self, df, channel_id, required=False):
        """Return a Series for a stable channel id, or None if unavailable."""
        column = self.resolve_channel(df, channel_id, required=required)
        if column:
            series = df[column]
            if self._is_shock_acceleration_channel(channel_id, column):
                return series / MM_PER_S2_PER_G
            return series
        derived = self._derived_channel_series(df, channel_id)
        if derived is not None:
            return derived
        return None

    def _is_shock_acceleration_channel(self, channel_id, column):
        """Identify shock acceleration channels that should be displayed in g."""
        shock_acc_channels = {'shock_acc_fl', 'shock_acc_fr', 'shock_acc_rl', 'shock_acc_rr'}
        if channel_id in shock_acc_channels:
            return True

        normalized_column = self._normalize_channel_name(column)
        return normalized_column.startswith('accelerationon') and 'shockpos' in normalized_column

    def _derived_channel_series(self, df, channel_id):
        """Return computed channels that are not present in the raw export."""
        shock_acc_sources = {
            'shock_acc_fl': 'shock_vel_fl',
            'shock_acc_fr': 'shock_vel_fr',
            'shock_acc_rl': 'shock_vel_rl',
            'shock_acc_rr': 'shock_vel_rr',
        }
        if channel_id not in shock_acc_sources:
            return None

        velocity = self.channel_series(df, shock_acc_sources[channel_id])
        time = self.channel_series(df, 'time')
        if velocity is None or time is None or len(velocity) < 2:
            return None

        time_delta = time.astype(float).diff().replace(0, np.nan)
        acceleration = velocity.astype(float).diff() / time_delta
        acceleration = acceleration.replace([np.inf, -np.inf], np.nan)
        return acceleration.fillna(0.0) / MM_PER_S2_PER_G

    def has_channels(self, df, channel_ids):
        """Return True if every channel id can be resolved in df."""
        return all(self.resolve_channel(df, channel_id) for channel_id in channel_ids)
    
    def extract_best_lap(self, df):
        """
        Extract best lap using segment times and beacon markers
        Beacon markers indicate end of each lap/segment
        """
        beacon_markers = df.attrs.get('beacon_markers', [])
        segment_times = df.attrs.get('segment_times', [])
        
        if not beacon_markers or not segment_times:
            print(f"  Warning: No segment time data found, using entire dataset")
            return df

        valid_segments = self.get_valid_lap_segments(df)
        if valid_segments:
            best_segment = min(valid_segments, key=lambda seg: seg['duration'])
            lap_data = self._extract_segment_data(df, best_segment)
            print(f"  Best lap time: {best_segment['duration']:.3f}s (segment {best_segment['index'] + 1})")
            if best_segment.get('distance') is not None:
                print(f"  Best lap distance: {best_segment['distance']:.1f}m")
            print(f"  Extracted {len(lap_data)} samples for best lap")
            return lap_data
        
        print(f"  Using entire dataset as fallback")
        return df
    
    def get_lap_segments(self, df):
        """Return lap segment start/end times from beacon markers and segment times."""
        beacon_markers = df.attrs.get('beacon_markers', [])
        segment_times = df.attrs.get('segment_times', [])
        
        if not beacon_markers or not segment_times:
            return []
        
        segments = []
        start_time = 0.0
        for duration, end_time in zip(segment_times, beacon_markers):
            segment = {
                'index': len(segments),
                'start': start_time,
                'end': end_time,
                'duration': duration
            }
            self._add_segment_metrics(df, segment)
            segments.append(segment)
            start_time = end_time
        return segments

    def _add_segment_metrics(self, df, segment):
        """Add sample, distance, and speed metrics used to reject incomplete laps."""
        lap_data = self._extract_segment_data(df, segment, reset_time=False, include_attrs=False)
        segment['samples'] = len(lap_data)

        distance_series = self.channel_series(lap_data, 'distance')
        if len(lap_data) and distance_series is not None:
            distance = distance_series.dropna()
            segment['distance'] = float(distance.iloc[-1] - distance.iloc[0]) if len(distance) >= 2 else 0.0
        else:
            segment['distance'] = None

        gps_speed = self.channel_series(lap_data, 'gps_speed')
        if len(lap_data) and gps_speed is not None:
            segment['max_speed'] = float(gps_speed.max())
        else:
            segment['max_speed'] = None

    def _duration_valid_segments(self, segments):
        """Return segments in the normal lap-time cluster."""
        durations = np.array([seg['duration'] for seg in segments if seg['duration'] > 0], dtype=float)
        if len(durations) == 0:
            return []

        typical_duration = float(np.median(durations))
        min_duration = typical_duration * 0.50
        max_duration = typical_duration * 1.50
        return [seg for seg in segments if min_duration <= seg['duration'] <= max_duration]

    def _distance_valid_segments(self, segments):
        """Return segments that cover a near-complete lap distance."""
        distances = [seg['distance'] for seg in segments if seg.get('distance') is not None and seg['distance'] > 0]
        if not distances:
            return segments

        reference_distance = max(distances)
        min_distance = reference_distance * 0.80
        return [seg for seg in segments if seg.get('distance') is not None and seg['distance'] >= min_distance]

    def get_valid_lap_segments(self, df):
        """Return complete, representative lap segments eligible for best-lap selection."""
        segments = [seg for seg in self.get_lap_segments(df) if seg['duration'] > 0]
        if not segments:
            return []

        distance_valid_segments = self._distance_valid_segments(segments)
        valid_segments = self._duration_valid_segments(distance_valid_segments)

        # Fallback for files without usable distance data: use duration and sample count only.
        if not valid_segments:
            valid_segments = self._duration_valid_segments(segments)

        valid_segments = [seg for seg in valid_segments if seg.get('samples', 0) >= 40]

        ignored = len(segments) - len(valid_segments)
        if ignored:
            print(f"  Ignored {ignored} incomplete/outlier segment(s)")
        return valid_segments

    def _extract_segment_data(self, df, segment, reset_time=True, include_attrs=True):
        """Extract a segment and optionally make Time relative to lap start."""
        time_column = self.resolve_channel(df, 'time')
        if not time_column:
            lap_data = df.copy()
        else:
            lap_data = df[(df[time_column] >= segment['start']) & (df[time_column] <= segment['end'])].copy()

        if reset_time and time_column and time_column in lap_data.columns:
            lap_data[time_column] = lap_data[time_column] - segment['start']

        if include_attrs:
            lap_data.attrs['lap_duration'] = segment['duration']
            lap_data.attrs['segment_index'] = segment['index']
            lap_data.attrs['segment_start'] = segment['start']
            lap_data.attrs['segment_end'] = segment['end']
            lap_data.attrs['lap_distance'] = segment.get('distance')
        return lap_data.reset_index(drop=True)
    
    def get_qualifying_laps(self, df, threshold_multiplier=1.2):
        """Return all laps within threshold_multiplier of the best lap."""
        segments = self.get_valid_lap_segments(df)
        if not segments:
            return [df]
        
        best_duration = min(seg['duration'] for seg in segments)
        threshold = best_duration * threshold_multiplier
        qualifying = []
        for seg in segments:
            if seg['duration'] <= threshold and seg['duration'] > 0:
                lap_data = self._extract_segment_data(df, seg)
                if len(lap_data) > 20:
                    qualifying.append(lap_data)
        
        if not qualifying:
            print(f"  No laps found within {threshold_multiplier*100:.0f}% of best lap; using full dataset")
            return [df]
        
        print(f"  Best lap = {best_duration:.3f}s, selected {len(qualifying)} lap(s) within {threshold_multiplier*100:.0f}%")
        return qualifying
    
    def _combine_laps(self, laps):
        """Concatenate qualifying lap data for plotting."""
        if not laps:
            return pd.DataFrame()
        return pd.concat(laps, ignore_index=True)

    def _selected_graphs(self):
        """Return configured graphs selected for this run."""
        all_graphs = DASHBOARD_GRAPHS + self.extra_graphs
        available_graphs = [graph for graph in all_graphs if graph.get('enabled', True)]
        if not self.graph_ids:
            return available_graphs

        graph_ids = set(self.graph_ids)
        graphs = [graph for graph in all_graphs if graph['id'] in graph_ids]
        missing = sorted(graph_ids - {graph['id'] for graph in all_graphs})
        if missing:
            raise ValueError(f"Unknown graph id(s): {', '.join(missing)}")
        return graphs

    def _record_plot(self, filename, title, section, mode):
        """Track generated plots for the HTML dashboard."""
        if not filename:
            return
        self.generated_plots.append({
            'filename': filename,
            'title': title,
            'section': section,
            'mode': mode,
        })

    def _renderer_for_graph(self, graph):
        """Create a renderer object for a configured graph."""
        return GraphRendererFactory.create(self, graph)

    def _write_dashboard_html(self, filename, page_title):
        """Write a simple static HTML dashboard for generated plots."""
        sections = []
        for plot in self.generated_plots:
            if plot['section'] not in sections:
                sections.append(plot['section'])

        cards = []
        for section in sections:
            section_plots = [plot for plot in self.generated_plots if plot['section'] == section]
            cards.append(f"<h2>{escape(section)}</h2>")
            cards.append('<div class="grid">')
            for plot in section_plots:
                title = escape(plot['title'])
                filename_escaped = escape(plot['filename'])
                cards.append(
                    f'<article class="card">'
                    f'<h3>{title}</h3>'
                    f'<a href="{filename_escaped}"><img src="{filename_escaped}" alt="{title}"></a>'
                    f'</article>'
                )
            cards.append('</div>')

        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(page_title)}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f6f8; color: #1b1f24; }}
    header {{ padding: 24px 32px; background: #151922; color: white; }}
    h1 {{ margin: 0; font-size: 28px; }}
    main {{ padding: 24px 32px 40px; }}
    h2 {{ margin: 28px 0 14px; font-size: 22px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #d8dde6; border-radius: 8px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
    .card h3 {{ margin: 0 0 10px; font-size: 16px; }}
    .card img {{ width: 100%; height: auto; display: block; border: 1px solid #edf0f4; }}
    code {{ background: #e9edf3; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header><h1>{escape(page_title)}</h1></header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
        output_path = self.data_dir / filename
        output_path.write_text(html, encoding='utf-8')
        print(f"  Saved dashboard: {filename}")
        return filename
    
    def analyze(self, mode='comparison'):
        """
        Analyze telemetry data
        mode: 'comparison' (aero vs no-aero) or 'individual' (single file analysis)
        """
        self.generated_plots = []
        self.load_all_files()
        
        # Extract best laps for each file
        for label, df in self.data.items():
            print(f"\nExtracting best lap from {label}...")
            self.best_laps[label] = self.extract_best_lap(df)
        
        if mode == 'comparison':
            self._analyze_comparison()
        elif mode == 'individual':
            self._analyze_individual()
        else:
            raise ValueError("mode must be 'comparison' or 'individual'")

        dashboard_name = f"dashboard_{mode}.html"
        self._write_dashboard_html(dashboard_name, f"{DASHBOARD_TITLE} - {mode.title()} Mode")
    
    def _analyze_individual(self):
        """Generate analysis plots for each file individually"""
        for label, lap_data in self.best_laps.items():
            config_label = self._get_config_label(label)
            print(f"\nGenerating plots for {label} ({config_label})...")
            self._generate_individual_dashboard_plots(label, lap_data, config_label)

    def _generate_individual_dashboard_plots(self, label, lap_data, config_label):
        """Generate selected dashboard plots for one run."""
        for graph in self._selected_graphs():
            renderer = self._renderer_for_graph(graph)
            filename = renderer.render_individual(label, lap_data, config_label)
            category = graph.get('category', 'General')
            section = f"Individual Run - {label} ({config_label}) - {category}"
            self._record_plot(filename, graph['name'], section, 'individual')
    
    def _get_config_label(self, label):
        """Determine if a dataset is aero or no-aero based on metadata"""
        if label in self.config_labels:
            return self.config_labels[label]
        if label in self.data:
            metadata = self.data[label].attrs.get('metadata', {})
            comment = metadata.get('Comment', '').lower()
            if 'no aero' in comment or 'no-aero' in comment or 'without aero' in comment:
                return 'No Aero'
            if 'aero' in comment:
                return 'Aero'
            else:
                return 'No Aero'
        return label
    
    def _find_comparison_groups(self):
        """Return no-aero and aero label groups for comparison."""
        no_aero_labels = []
        aero_labels = []
        for label in sorted(self.data.keys()):
            config = self._get_config_label(label)
            if config == 'Aero':
                aero_labels.append(label)
            elif config == 'No Aero':
                no_aero_labels.append(label)
        return no_aero_labels, aero_labels

    def _tag_lap(self, lap_data, source_label):
        """Attach source metadata to an extracted lap."""
        lap_data.attrs['source_label'] = source_label
        if 'Source File' not in lap_data.columns:
            lap_data = lap_data.copy()
            lap_data['Source File'] = source_label
        return lap_data

    def _best_lap_for_group(self, labels):
        """Return the fastest valid complete lap across a group of files."""
        best_label = None
        best_segment = None
        for label in labels:
            valid_segments = self.get_valid_lap_segments(self.data[label])
            if not valid_segments:
                continue
            candidate = min(valid_segments, key=lambda seg: seg['duration'])
            if best_segment is None or candidate['duration'] < best_segment['duration']:
                best_label = label
                best_segment = candidate

        if best_label is None:
            return []

        lap_data = self._extract_segment_data(self.data[best_label], best_segment)
        lap_data = self._tag_lap(lap_data, best_label)
        print(f"  Best group lap: {best_label} lap {best_segment['index'] + 1} ({best_segment['duration']:.3f}s)")
        return [lap_data]

    def _laps_within_best_threshold_for_group(self, labels, threshold_multiplier=1.2):
        """Return all valid complete laps within threshold_multiplier of each file's best lap."""
        qualifying_laps = []
        for label in labels:
            valid_segments = self.get_valid_lap_segments(self.data[label])
            if not valid_segments:
                continue
            best_duration = min(seg['duration'] for seg in valid_segments)
            threshold = best_duration * threshold_multiplier
            selected = [seg for seg in valid_segments if seg['duration'] <= threshold]
            print(f"  {label}: selected {len(selected)} lap(s) <= {threshold_multiplier*100:.0f}% of {best_duration:.3f}s best")
            for seg in selected:
                lap_data = self._extract_segment_data(self.data[label], seg)
                lap_data = self._tag_lap(lap_data, label)
                qualifying_laps.append(lap_data)
        return qualifying_laps
    
    def _analyze_comparison(self):
        """Generate comparison plots for aero vs no-aero"""
        no_aero_labels, aero_labels = self._find_comparison_groups()
        if not no_aero_labels or not aero_labels:
            print("Need one aero and one no aero file for comparison. Running individual analysis instead...")
            self._analyze_individual()
            return

        no_aero_label = f"No Aero ({'+'.join(no_aero_labels)})"
        aero_label = f"Aero ({'+'.join(aero_labels)})"

        if 'best' in self.comparison_lap_sets:
            print(f"\nComparing best laps: {no_aero_label} vs {aero_label}")
            best_no_aero_laps = self._best_lap_for_group(no_aero_labels)
            best_aero_laps = self._best_lap_for_group(aero_labels)
            self._generate_comparison_plots(
                best_no_aero_laps,
                best_aero_laps,
                no_aero_label,
                aero_label,
                plot_context='Best Laps Only',
                filename_suffix='best_lap',
            )

        threshold_percent = self.lap_threshold * 100
        threshold_slug = f"{int(round(threshold_percent))}pct"
        if 'all' in self.comparison_lap_sets:
            print(f"\nComparing all laps within {threshold_percent:.0f}% of each file's best lap: {no_aero_label} vs {aero_label}")
            all_no_aero_laps = self._laps_within_best_threshold_for_group(no_aero_labels, threshold_multiplier=self.lap_threshold)
            all_aero_laps = self._laps_within_best_threshold_for_group(aero_labels, threshold_multiplier=self.lap_threshold)
            self._generate_comparison_plots(
                all_no_aero_laps,
                all_aero_laps,
                no_aero_label,
                aero_label,
                plot_context=f'All Laps Within {threshold_percent:.0f}% of Best Lap',
                filename_suffix=f'all_laps_{threshold_slug}',
            )
    
    def _generate_comparison_plots(self, no_aero_laps, aero_laps, no_aero_label='No Aero', aero_label='Aero', plot_context='Best Laps Only', filename_suffix='best_lap'):
        """Generate one set of overlayed comparison plots."""
        for graph in self._selected_graphs():
            renderer = self._renderer_for_graph(graph)
            filename = renderer.render_comparison(no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix)
            category = graph.get('category', 'General')
            self._record_plot(filename, graph['name'], f"{plot_context} - {category}", 'comparison')
    
    def _generate_all_plots(self, primary_label, primary_data, comparison_data, primary_config='Primary', comparison_config='Comparison'):
        """Generate all required visualizations"""
        
        # 1. Shock Travel & Velocity Histograms
        self._plot_shock_histograms(primary_label, primary_data, comparison_data, primary_config, comparison_config)
        
        # 2. GPS Speed & Yaw Rate & ECU RPM
        self._plot_speed_yaw_rpm(primary_label, primary_data, comparison_data, primary_config, comparison_config)
        
        # 3. GG Diagram
        self._plot_gg_diagram(primary_label, primary_data, comparison_data, primary_config, comparison_config)
        
        # 4. GPS Speed vs Yaw Rate
        self._plot_speed_vs_yaw(primary_label, primary_data, comparison_data, primary_config, comparison_config)
        
        # 5. GPS LatAcc vs Yaw Rate
        self._plot_latacc_vs_yaw(primary_label, primary_data, comparison_data, primary_config, comparison_config)
        
        # 6. GPS Speed vs Wheel Speed (RL & RR)
        self._plot_speed_vs_wheelspeed(primary_label, primary_data, comparison_data, primary_config, comparison_config)

    def _hist_percent(self, series, bins):
        """Return histogram bin percentages for a numeric series."""
        clean = series.dropna()
        if clean.empty:
            return np.array([]), np.array([])
        counts, edges = np.histogram(clean, bins=bins)
        percentages = counts / counts.sum() * 100 if counts.sum() else counts
        return percentages, edges

    def _combined_bins(self, series_list, bins=10):
        """Build shared bins across one or more numeric series."""
        clean_series = [series.dropna() for series in series_list if series is not None and not series.dropna().empty]
        if not clean_series:
            return bins
        combined = pd.concat(clean_series, ignore_index=True)
        lower = combined.min()
        upper = combined.max()
        if lower == upper:
            lower -= 0.5
            upper += 0.5
        return np.linspace(lower, upper, bins + 1)

    def _draw_percent_bars(self, ax, series, bins, color, label, alpha=0.65, hatch=None, annotate=True):
        """Draw a percent histogram with optional percentage labels."""
        percentages, edges = self._hist_percent(series, bins)
        if len(percentages) == 0:
            return

        widths = np.diff(edges)
        bars = ax.bar(
            edges[:-1],
            percentages,
            width=widths,
            align='edge',
            color=color,
            alpha=alpha,
            edgecolor=color if hatch else 'black',
            linewidth=1.0,
            hatch=hatch,
            label=label,
        )

        if annotate:
            for bar, percent in zip(bars, percentages):
                if percent < 2.0:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'{percent:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color='#1f2a44',
                    bbox={'facecolor': 'white', 'edgecolor': '#d9d9d9', 'alpha': 0.85, 'pad': 1.2},
                )

    def _slugify(self, value):
        """Return a filesystem-safe slug."""
        slug = ''.join(char.lower() if char.isalnum() else '_' for char in str(value))
        return '_'.join(part for part in slug.split('_') if part)

    def _plot_configurable_individual(self, graph, label, data, config_label):
        """Render a graph defined entirely in dashboard_config.py for one run."""
        kind = graph.get('kind')
        title = f"{graph['name']} - {label}"
        filename = f"{self._slugify(graph['id'])}_{label}.png"

        if kind == 'timeseries':
            fig, ax = plt.subplots(figsize=(12, 6))
            x_channel = graph.get('x', 'time')
            x = self.channel_series(data, x_channel)
            if x is None:
                print(f"  Skipped {graph['name']} for {label}: missing x channel {x_channel}")
                return None
            plotted = False
            for channel in graph.get('channels', []):
                y = self.channel_series(data, channel['id'])
                if y is None:
                    continue
                ax.plot(x, y, linewidth=1.2, alpha=0.85, label=channel.get('label', channel['id']))
                plotted = True
            if not plotted:
                print(f"  Skipped {graph['name']} for {label}: no configured channels found")
                plt.close()
                return None
            ax.set_xlabel(graph.get('x_label', x_channel))
            ax.set_ylabel(graph.get('y_label', graph['name']))
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

        elif kind == 'scatter':
            x = self.channel_series(data, graph['x'])
            y = self.channel_series(data, graph['y'])
            if x is None or y is None:
                print(f"  Skipped {graph['name']} for {label}: missing x/y channel")
                return None
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.scatter(x, y, alpha=0.5, s=22, label=config_label)
            ax.set_xlabel(graph.get('x_label', graph['x']))
            ax.set_ylabel(graph.get('y_label', graph['y']))
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

        elif kind == 'histogram_percent':
            channels = graph.get('channels', [])
            fig, axes = plt.subplots(len(channels), 1, figsize=(12, max(4, 3.5 * len(channels))))
            axes = np.atleast_1d(axes)
            plotted = False
            for ax, channel in zip(axes, channels):
                series = self.channel_series(data, channel['id'])
                if series is None:
                    ax.set_visible(False)
                    continue
                bins = self._combined_bins([series], bins=graph.get('bins', 12))
                self._draw_percent_bars(ax, series, bins, '#1f77b4', channel.get('label', channel['id']))
                ax.set_xlabel(channel.get('label', channel['id']))
                ax.set_ylabel('Percent [%]')
                ax.grid(True, alpha=0.3)
                ax.legend()
                plotted = True
            if not plotted:
                print(f"  Skipped {graph['name']} for {label}: no configured channels found")
                plt.close()
                return None
            plt.suptitle(title, fontsize=14, fontweight='bold')

        else:
            print(f"  Skipped {graph['name']}: unknown graph kind '{kind}'")
            return None

        plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
        return filename

    def _plot_configurable_comparison(self, graph, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        """Render a graph defined entirely in dashboard_config.py for comparison mode."""
        kind = graph.get('kind')
        title = f"{graph['name']} - {plot_context}"
        filename = f"{self._slugify(graph['id'])}_{filename_suffix}_comparison.png"
        no_aero_data = self._combine_laps(no_aero_laps)
        aero_data = self._combine_laps(aero_laps)

        if kind == 'timeseries':
            fig, ax = plt.subplots(figsize=(12, 6))
            x_channel = graph.get('x', 'time')
            plotted = False
            for laps, label_text, color in [(no_aero_laps, no_aero_label, '#1f77b4'), (aero_laps, aero_label, '#d62728')]:
                first_line = True
                for lap in laps:
                    x = self.channel_series(lap, x_channel)
                    if x is None:
                        continue
                    for channel in graph.get('channels', []):
                        y = self.channel_series(lap, channel['id'])
                        if y is None:
                            continue
                        line_label = f"{label_text} {channel.get('label', channel['id'])}" if first_line else None
                        ax.plot(x, y, linewidth=1.0, alpha=0.45, color=color, label=line_label)
                        plotted = True
                    first_line = False
            if not plotted:
                print(f"  Skipped {graph['name']}: no configured channels found")
                plt.close()
                return None
            ax.set_xlabel(graph.get('x_label', x_channel))
            ax.set_ylabel(graph.get('y_label', graph['name']))

        elif kind == 'scatter':
            fig, ax = plt.subplots(figsize=(10, 7))
            plotted = False
            for data, label_text, color in [(no_aero_data, no_aero_label, '#1f77b4'), (aero_data, aero_label, '#d62728')]:
                x = self.channel_series(data, graph['x'])
                y = self.channel_series(data, graph['y'])
                if x is None or y is None:
                    continue
                ax.scatter(x, y, alpha=0.4, s=22, label=label_text, color=color)
                plotted = True
            if not plotted:
                print(f"  Skipped {graph['name']}: missing x/y channel")
                plt.close()
                return None
            ax.set_xlabel(graph.get('x_label', graph['x']))
            ax.set_ylabel(graph.get('y_label', graph['y']))

        elif kind == 'histogram_percent':
            channels = graph.get('channels', [])
            fig, axes = plt.subplots(len(channels), 1, figsize=(12, max(4, 3.5 * len(channels))))
            axes = np.atleast_1d(axes)
            plotted = False
            for ax, channel in zip(axes, channels):
                no_aero_series = self.channel_series(no_aero_data, channel['id'])
                aero_series = self.channel_series(aero_data, channel['id'])
                bins = self._combined_bins([no_aero_series, aero_series], bins=graph.get('bins', 12))
                if no_aero_series is not None:
                    self._draw_percent_bars(ax, no_aero_series, bins, '#1f77b4', no_aero_label, alpha=0.55)
                    plotted = True
                if aero_series is not None:
                    self._draw_percent_bars(ax, aero_series, bins, '#d62728', aero_label, alpha=0.30, hatch='///', annotate=False)
                    plotted = True
                ax.set_xlabel(channel.get('label', channel['id']))
                ax.set_ylabel('Percent [%]')
                ax.grid(True, alpha=0.3)
                ax.legend()
            if not plotted:
                print(f"  Skipped {graph['name']}: no configured channels found")
                plt.close()
                return None
            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filename}")
            plt.close()
            return filename

        else:
            print(f"  Skipped {graph['name']}: unknown graph kind '{kind}'")
            return None

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
        return filename
    
    def _plot_shock_histograms(self, label, data, comparison_data=None, primary_config='Primary', comparison_config='Comparison'):
        """Plot shock travel and velocity percent histograms in tiled configuration."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.subplots_adjust(top=0.86, hspace=0.55, wspace=0.20)
        shocks = [
            ('shock_pos_fl', 'shock_vel_fl', 'FL'),
            ('shock_pos_fr', 'shock_vel_fr', 'FR'),
            ('shock_pos_rl', 'shock_vel_rl', 'RL'),
            ('shock_pos_rr', 'shock_vel_rr', 'RR')
        ]
        shock_colors = {'FL': '#e41a1c', 'FR': '#00c800', 'RL': '#0000ff', 'RR': '#ff7f0e'}

        for idx, (pos_col, vel_col, shock_name) in enumerate(shocks):
            ax = axes[idx // 2, idx % 2]
            color = shock_colors[shock_name]

            pos_series = self.channel_series(data, pos_col)
            if pos_series is not None:
                self._draw_percent_bars(ax, pos_series, self._combined_bins([pos_series], bins=10), color, f'{primary_config} travel')

            ax.set_title(f'{shock_name} Shock Travel & Velocity', fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{shock_name} Shock Pos [mm]')
            ax.set_ylabel('Percent [%]')
            ax.grid(True, alpha=0.25)

            vel_series = self.channel_series(data, vel_col)
            if vel_series is not None:
                vel_ax = ax.twiny()
                percentages, edges = self._hist_percent(vel_series, self._combined_bins([vel_series], bins=10))
                centers = edges[:-1] + np.diff(edges) / 2
                vel_ax.step(centers, percentages, where='mid', color='black', linewidth=1.6, label=f'{primary_config} velocity')
                vel_ax.set_xlabel(f'Velocity on {shock_name} Shock Pos [mm/s]')
                vel_ax.tick_params(axis='x', labelsize=8)

            ax.legend(fontsize=8, loc='upper left')

        plt.suptitle(f'Shock Travel & Velocity Percent Histograms - {label}', fontsize=16, fontweight='bold')
        filename = f'shock_analysis_{label}.png'
        plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
        return filename
    
    def _plot_shock_histograms_comparison(self, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        """Plot shock travel and velocity percent histograms for aero vs no aero."""
        no_aero_data = self._combine_laps(no_aero_laps)
        aero_data = self._combine_laps(aero_laps)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.subplots_adjust(top=0.86, hspace=0.55, wspace=0.20)
        shocks = [
            ('shock_pos_fl', 'shock_vel_fl', 'FL'),
            ('shock_pos_fr', 'shock_vel_fr', 'FR'),
            ('shock_pos_rl', 'shock_vel_rl', 'RL'),
            ('shock_pos_rr', 'shock_vel_rr', 'RR')
        ]
        shock_colors = {'FL': '#e41a1c', 'FR': '#00c800', 'RL': '#0000ff', 'RR': '#ff7f0e'}

        for idx, (pos_col, vel_col, shock_name) in enumerate(shocks):
            ax = axes[idx // 2, idx % 2]
            color = shock_colors[shock_name]

            no_aero_pos = self.channel_series(no_aero_data, pos_col)
            aero_pos = self.channel_series(aero_data, pos_col)
            pos_series = [no_aero_pos, aero_pos]
            pos_bins = self._combined_bins(pos_series, bins=10)
            if no_aero_pos is not None:
                self._draw_percent_bars(ax, no_aero_pos, pos_bins, color, f'{no_aero_label} travel', alpha=0.45)
            if aero_pos is not None:
                self._draw_percent_bars(ax, aero_pos, pos_bins, color, f'{aero_label} travel', alpha=0.22, hatch='///', annotate=False)

            ax.set_title(f'{shock_name} Shock Travel & Velocity', fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{shock_name} Shock Pos [mm]')
            ax.set_ylabel('Percent [%]')
            ax.grid(True, alpha=0.25)

            no_aero_vel = self.channel_series(no_aero_data, vel_col)
            aero_vel = self.channel_series(aero_data, vel_col)
            vel_series = [no_aero_vel, aero_vel]
            vel_bins = self._combined_bins(vel_series, bins=10)
            vel_ax = ax.twiny()
            if no_aero_vel is not None:
                percentages, edges = self._hist_percent(no_aero_vel, vel_bins)
                centers = edges[:-1] + np.diff(edges) / 2
                vel_ax.step(centers, percentages, where='mid', color='black', linewidth=1.6, label=f'{no_aero_label} velocity')
            if aero_vel is not None:
                percentages, edges = self._hist_percent(aero_vel, vel_bins)
                centers = edges[:-1] + np.diff(edges) / 2
                vel_ax.step(centers, percentages, where='mid', color='black', linewidth=1.6, linestyle='--', label=f'{aero_label} velocity')
            vel_ax.set_xlabel(f'Velocity on {shock_name} Shock Pos [mm/s]')
            vel_ax.tick_params(axis='x', labelsize=8)

            handles, labels = ax.get_legend_handles_labels()
            vel_handles, vel_labels = vel_ax.get_legend_handles_labels()
            ax.legend(handles + vel_handles, labels + vel_labels, fontsize=7, loc='upper left')

        plt.suptitle(f'Testing #2 at Amigo Track 2 - Shock Travel & Velocity Percent Comparison - {plot_context}', fontsize=16, fontweight='bold')
        filename = f'shock_analysis_{filename_suffix}_comparison.png'
        plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
        return filename
    
    def _plot_speed_yaw_rpm_comparison(self, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        """Plot GPS speed, yaw rate, and ECU RPM for aero vs no aero."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        for idx, lap in enumerate(no_aero_laps):
            time = self.channel_series(lap, 'time')
            gps_speed = self.channel_series(lap, 'gps_speed')
            yaw_rate = self.channel_series(lap, 'yaw_rate')
            rpm = self.channel_series(lap, 'ecu_rpm')
            if time is not None and gps_speed is not None:
                axes[0].plot(time, gps_speed, label=no_aero_label if idx == 0 else None, linewidth=1.0, alpha=0.45, color='blue')
            if time is not None and yaw_rate is not None:
                axes[1].plot(time, yaw_rate, label=no_aero_label if idx == 0 else None, linewidth=1.0, alpha=0.45, color='green')
            if time is not None and rpm is not None:
                axes[2].plot(time, rpm, label=no_aero_label if idx == 0 else None, linewidth=1.0, alpha=0.45, color='red')
        for idx, lap in enumerate(aero_laps):
            time = self.channel_series(lap, 'time')
            gps_speed = self.channel_series(lap, 'gps_speed')
            yaw_rate = self.channel_series(lap, 'yaw_rate')
            rpm = self.channel_series(lap, 'ecu_rpm')
            if time is not None and gps_speed is not None:
                axes[0].plot(time, gps_speed, label=aero_label if idx == 0 else None, linewidth=1.0, alpha=0.45, color='red')
            if time is not None and yaw_rate is not None:
                axes[1].plot(time, yaw_rate, label=aero_label if idx == 0 else None, linewidth=1.0, alpha=0.45, color='orange')
            if time is not None and rpm is not None:
                axes[2].plot(time, rpm, label=aero_label if idx == 0 else None, linewidth=1.0, alpha=0.45, color='purple')
        axes[0].set_ylabel('GPS Speed (km/h)', fontsize=11)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].set_ylabel('Yaw Rate (deg/s)', fontsize=11)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[2].set_ylabel('ECU RPM', fontsize=11)
        axes[2].set_xlabel('Time (s)', fontsize=11)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        plt.suptitle(f'Testing #2 at Amigo Track 2 - GPS Speed, Yaw Rate & RPM Comparison - {plot_context}', fontsize=14, fontweight='bold')
        filename = f'speed_yaw_rpm_{filename_suffix}_comparison.png'
        plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
        return filename

    def _plot_speed_yaw_rpm_histograms_comparison(self, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        """Plot overlaid histograms for GPS speed, yaw rate, and ECU RPM."""
        no_aero_data = self._combine_laps(no_aero_laps)
        aero_data = self._combine_laps(aero_laps)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        metrics = [
            ('gps_speed', 'GPS Speed (km/h)'),
            ('yaw_rate', 'Yaw Rate (deg/s)'),
            ('ecu_rpm', 'ECU RPM'),
        ]
        colors = {
            no_aero_label: '#1f77b4',
            aero_label: '#d62728',
        }

        for ax, (column, xlabel) in zip(axes, metrics):
            plotted = False
            bins = self._combined_bins([
                self.channel_series(no_aero_data, column),
                self.channel_series(aero_data, column),
            ], bins=12)
            no_aero_series = self.channel_series(no_aero_data, column)
            aero_series = self.channel_series(aero_data, column)
            if no_aero_series is not None:
                self._draw_percent_bars(ax, no_aero_series, bins, colors[no_aero_label], no_aero_label, alpha=0.55)
                plotted = True
            if aero_series is not None:
                self._draw_percent_bars(ax, aero_series, bins, colors[aero_label], aero_label, alpha=0.30, hatch='///', annotate=False)
                plotted = True
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel('Percent [%]', fontsize=11)
            ax.grid(True, alpha=0.3)
            if plotted:
                ax.legend()

        plt.suptitle(f'Testing #2 at Amigo Track 2 - GPS Speed, Yaw Rate & RPM Histograms - {plot_context}', fontsize=14, fontweight='bold')
        filename = f'speed_yaw_rpm_histograms_{filename_suffix}_comparison.png'
        plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
        return filename
    
    def _plot_gg_diagram_comparison(self, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        """Plot the GG diagram for aero vs no aero."""
        no_aero_data = self._combine_laps(no_aero_laps)
        aero_data = self._combine_laps(aero_laps)
        fig, ax = plt.subplots(figsize=(10, 10))
        no_aero_lat = self.channel_series(no_aero_data, 'gps_lat_acc')
        no_aero_lon = self.channel_series(no_aero_data, 'gps_lon_acc')
        aero_lat = self.channel_series(aero_data, 'gps_lat_acc')
        aero_lon = self.channel_series(aero_data, 'gps_lon_acc')
        if no_aero_lat is not None and no_aero_lon is not None:
            ax.scatter(no_aero_lat, no_aero_lon, alpha=0.5, s=20, label=no_aero_label, color='blue')
        if aero_lat is not None and aero_lon is not None:
            ax.scatter(aero_lat, aero_lon, alpha=0.5, s=20, label=aero_label, color='red')
        ax.set_xlabel('GPS LatAcc (g)', fontsize=12)
        ax.set_ylabel('GPS LonAcc (g)', fontsize=12)
        ax.set_title(f'Testing #2 at Amigo Track 2 - GG Diagram Comparison - {plot_context}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        for radius in [0.5, 1.0, 1.5, 2.0]:
            circle = plt.Circle((0, 0), radius, fill=False, linestyle='--', color='gray', alpha=0.3, linewidth=1)
            ax.add_patch(circle)
        filename = f'gg_diagram_{filename_suffix}_comparison.png'
        plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
        return filename
    
    def _plot_speed_vs_yaw_comparison(self, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        """Plot GPS Speed vs Yaw Rate comparison."""
        no_aero_data = self._combine_laps(no_aero_laps)
        aero_data = self._combine_laps(aero_laps)
        fig, ax = plt.subplots(figsize=(10, 7))
        no_aero_speed = self.channel_series(no_aero_data, 'gps_speed')
        no_aero_yaw = self.channel_series(no_aero_data, 'yaw_rate')
        aero_speed = self.channel_series(aero_data, 'gps_speed')
        aero_yaw = self.channel_series(aero_data, 'yaw_rate')
        if no_aero_speed is not None and no_aero_yaw is not None:
            ax.scatter(no_aero_yaw, no_aero_speed, alpha=0.4, s=25, label=no_aero_label, color='blue')
        if aero_speed is not None and aero_yaw is not None:
            ax.scatter(aero_yaw, aero_speed, alpha=0.4, s=25, label=aero_label, color='red')
        ax.set_xlabel('Yaw Rate (deg/s)', fontsize=12)
        ax.set_ylabel('GPS Speed (km/h)', fontsize=12)
        ax.set_title(f'Testing #2 at Amigo Track 2 - GPS Speed vs Yaw Rate - {plot_context}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        filename = f'speed_vs_yaw_{filename_suffix}_comparison.png'
        plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
        return filename
    
    def _plot_latacc_vs_yaw_comparison(self, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        """Plot GPS LatAcc vs Yaw Rate comparison."""
        no_aero_data = self._combine_laps(no_aero_laps)
        aero_data = self._combine_laps(aero_laps)
        fig, ax = plt.subplots(figsize=(10, 7))
        no_aero_lat = self.channel_series(no_aero_data, 'gps_lat_acc')
        no_aero_yaw = self.channel_series(no_aero_data, 'yaw_rate')
        aero_lat = self.channel_series(aero_data, 'gps_lat_acc')
        aero_yaw = self.channel_series(aero_data, 'yaw_rate')
        if no_aero_lat is not None and no_aero_yaw is not None:
            ax.scatter(no_aero_yaw, no_aero_lat, alpha=0.4, s=25, label=no_aero_label, color='green')
        if aero_lat is not None and aero_yaw is not None:
            ax.scatter(aero_yaw, aero_lat, alpha=0.4, s=25, label=aero_label, color='orange')
        ax.set_xlabel('Yaw Rate (deg/s)', fontsize=12)
        ax.set_ylabel('GPS LatAcc (g)', fontsize=12)
        ax.set_title(f'Testing #2 at Amigo Track 2 - GPS LatAcc vs Yaw Rate - {plot_context}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        filename = f'latacc_vs_yaw_{filename_suffix}_comparison.png'
        plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
        return filename
    
    def _plot_speed_vs_wheelspeed_comparison(self, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        """Plot GPS Speed vs wheel speeds comparison."""
        no_aero_data = self._combine_laps(no_aero_laps)
        aero_data = self._combine_laps(aero_laps)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        no_aero_speed = self.channel_series(no_aero_data, 'gps_speed')
        aero_speed = self.channel_series(aero_data, 'gps_speed')
        no_aero_rl = self.channel_series(no_aero_data, 'wheel_speed_rl')
        aero_rl = self.channel_series(aero_data, 'wheel_speed_rl')
        no_aero_rr = self.channel_series(no_aero_data, 'wheel_speed_rr')
        aero_rr = self.channel_series(aero_data, 'wheel_speed_rr')
        if no_aero_speed is not None and no_aero_rl is not None:
            axes[0].scatter(no_aero_speed, no_aero_rl, alpha=0.4, s=25, label=no_aero_label, color='blue')
        if aero_speed is not None and aero_rl is not None:
            axes[0].scatter(aero_speed, aero_rl, alpha=0.4, s=25, label=aero_label, color='red')
        axes[0].set_xlabel('GPS Speed (km/h)', fontsize=11)
        axes[0].set_ylabel('RL Wheel Speed (km/h)', fontsize=11)
        axes[0].set_title('GPS Speed vs RL Wheel Speed', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        if no_aero_speed is not None and no_aero_rr is not None:
            axes[1].scatter(no_aero_speed, no_aero_rr, alpha=0.4, s=25, label=no_aero_label, color='green')
        if aero_speed is not None and aero_rr is not None:
            axes[1].scatter(aero_speed, aero_rr, alpha=0.4, s=25, label=aero_label, color='orange')
        axes[1].set_xlabel('GPS Speed (km/h)', fontsize=11)
        axes[1].set_ylabel('RR Wheel Speed (km/h)', fontsize=11)
        axes[1].set_title('GPS Speed vs RR Wheel Speed', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        plt.suptitle(f'Testing #2 at Amigo Track 2 - GPS Speed vs Wheel Speeds Comparison - {plot_context}', fontsize=14, fontweight='bold')
        filename = f'speed_vs_wheelspeed_{filename_suffix}_comparison.png'
        plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
        return filename

    def _latlon_to_local_xy(self, data, origin_lat, origin_lon):
        """Convert GPS latitude/longitude degrees into local meters."""
        lat_series = self.channel_series(data, 'gps_latitude', required=True)
        lon_series = self.channel_series(data, 'gps_longitude', required=True)
        lat = np.deg2rad(lat_series)
        lon = np.deg2rad(lon_series)
        origin_lat_rad = np.deg2rad(origin_lat)
        origin_lon_rad = np.deg2rad(origin_lon)
        earth_radius_m = 6371000.0
        x = (lon - origin_lon_rad) * np.cos(origin_lat_rad) * earth_radius_m
        y = (lat - origin_lat_rad) * earth_radius_m
        return x, y

    def _plot_track_map_comparison(self, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        """Plot GPS track map lines for best aero and no-aero laps."""
        if not all(self.has_channels(lap, ['gps_latitude', 'gps_longitude']) for lap in no_aero_laps + aero_laps):
            print("  Skipped track map: GPS Latitude/GPS Longitude columns not found")
            return

        no_aero_gps_laps = [lap[[self.resolve_channel(lap, 'gps_latitude'), self.resolve_channel(lap, 'gps_longitude')]].dropna() for lap in no_aero_laps]
        aero_gps_laps = [lap[[self.resolve_channel(lap, 'gps_latitude'), self.resolve_channel(lap, 'gps_longitude')]].dropna() for lap in aero_laps]
        no_aero_gps_laps = [lap for lap in no_aero_gps_laps if not lap.empty]
        aero_gps_laps = [lap for lap in aero_gps_laps if not lap.empty]
        if not no_aero_gps_laps or not aero_gps_laps:
            print("  Skipped track map: no valid GPS coordinates")
            return

        origin_lat = pd.concat([self.channel_series(lap, 'gps_latitude') for lap in no_aero_gps_laps + aero_gps_laps], ignore_index=True).mean()
        origin_lon = pd.concat([self.channel_series(lap, 'gps_longitude') for lap in no_aero_gps_laps + aero_gps_laps], ignore_index=True).mean()

        fig, ax = plt.subplots(figsize=(10, 10))
        first_no_aero = True
        first_aero = True
        for gps_lap in no_aero_gps_laps:
            x, y = self._latlon_to_local_xy(gps_lap, origin_lat, origin_lon)
            ax.plot(x, y, label=no_aero_label if first_no_aero else None, color='#1f77b4', linewidth=1.8, alpha=0.55)
            if first_no_aero:
                ax.scatter(x.iloc[0], y.iloc[0], color='#1f77b4', marker='o', s=70, label=f'{no_aero_label} start')
            first_no_aero = False
        for gps_lap in aero_gps_laps:
            x, y = self._latlon_to_local_xy(gps_lap, origin_lat, origin_lon)
            ax.plot(x, y, label=aero_label if first_aero else None, color='#d62728', linewidth=1.8, alpha=0.55)
            if first_aero:
                ax.scatter(x.iloc[0], y.iloc[0], color='#d62728', marker='^', s=80, label=f'{aero_label} start')
            first_aero = False
        ax.set_title(f'Testing #2 at Amigo Track 2 - Track Map - {plot_context}', fontsize=14, fontweight='bold')
        ax.set_xlabel('East/West Position (m)', fontsize=12)
        ax.set_ylabel('North/South Position (m)', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.legend()
        filename = f'track_map_{filename_suffix}_comparison.png'
        plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
        return filename

    def _plot_track_map_individual(self, label, data, comparison_data=None, primary_config='Primary', comparison_config='Comparison'):
        """Plot a GPS track map for one best lap."""
        if not self.has_channels(data, ['gps_latitude', 'gps_longitude']):
            print(f"  Skipped track map for {label}: GPS Latitude/GPS Longitude columns not found")
            return None

        gps_data = data[[self.resolve_channel(data, 'gps_latitude'), self.resolve_channel(data, 'gps_longitude')]].dropna()
        if gps_data.empty:
            print(f"  Skipped track map for {label}: no valid GPS coordinates")
            return None

        origin_lat = self.channel_series(gps_data, 'gps_latitude').mean()
        origin_lon = self.channel_series(gps_data, 'gps_longitude').mean()
        x, y = self._latlon_to_local_xy(gps_data, origin_lat, origin_lon)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(x, y, label=primary_config, color='#1f77b4', linewidth=2.0)
        ax.scatter(x.iloc[0], y.iloc[0], color='#1f77b4', marker='o', s=70, label='start')
        ax.set_title(f'GPS Track Map - {label}', fontsize=14, fontweight='bold')
        ax.set_xlabel('East/West Position (m)', fontsize=12)
        ax.set_ylabel('North/South Position (m)', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.legend()
        filename = f'track_map_{label}.png'
        plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
        return filename
    
    def _plot_speed_yaw_rpm(self, label, data, comparison_data=None, primary_config='Primary', comparison_config='Comparison'):
        """Plot GPS speed with yaw rate and ECU RPM"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # GPS Speed
        time = self.channel_series(data, 'time')
        gps_speed = self.channel_series(data, 'gps_speed')
        if time is not None and gps_speed is not None:
            axes[0].plot(time, gps_speed, label=primary_config, linewidth=1.5, alpha=0.8)
            comparison_time = self.channel_series(comparison_data, 'time') if comparison_data is not None else None
            comparison_speed = self.channel_series(comparison_data, 'gps_speed') if comparison_data is not None else None
            if comparison_time is not None and comparison_speed is not None:
                axes[0].plot(comparison_time, comparison_speed, label=comparison_config, linewidth=1.5, alpha=0.8)
            axes[0].set_ylabel('GPS Speed (km/h)', fontsize=11)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Yaw Rate
        yaw_rate = self.channel_series(data, 'yaw_rate')
        if time is not None and yaw_rate is not None:
            axes[1].plot(time, yaw_rate, label=primary_config, linewidth=1.5, alpha=0.8, color='green')
            comparison_time = self.channel_series(comparison_data, 'time') if comparison_data is not None else None
            comparison_yaw = self.channel_series(comparison_data, 'yaw_rate') if comparison_data is not None else None
            if comparison_time is not None and comparison_yaw is not None:
                axes[1].plot(comparison_time, comparison_yaw, label=comparison_config, linewidth=1.5, alpha=0.8, color='orange')
            axes[1].set_ylabel('Yaw Rate (deg/s)', fontsize=11)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # ECU RPM
        rpm = self.channel_series(data, 'ecu_rpm')
        if time is not None and rpm is not None:
            axes[2].plot(time, rpm, label=primary_config, linewidth=1.5, alpha=0.8, color='red')
            comparison_time = self.channel_series(comparison_data, 'time') if comparison_data is not None else None
            comparison_rpm = self.channel_series(comparison_data, 'ecu_rpm') if comparison_data is not None else None
            if comparison_time is not None and comparison_rpm is not None:
                axes[2].plot(comparison_time, comparison_rpm, label=comparison_config, linewidth=1.5, alpha=0.8, color='purple')
            axes[2].set_ylabel('ECU RPM', fontsize=11)
            axes[2].set_xlabel('Time (s)', fontsize=11)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'GPS Speed, Yaw Rate & RPM Analysis - {label}', fontsize=14, fontweight='bold')
        filename = f'speed_yaw_rpm_{label}.png'
        plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
        return filename
    
    def _plot_gg_diagram(self, label, data, comparison_data=None, primary_config='Primary', comparison_config='Comparison'):
        """Plot GG diagram: GPS LonAcc vs GPS LatAcc"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        lat_acc = self.channel_series(data, 'gps_lat_acc')
        lon_acc = self.channel_series(data, 'gps_lon_acc')
        if lat_acc is not None and lon_acc is not None:
            ax.scatter(lat_acc, lon_acc, 
                      alpha=0.5, s=20, label=primary_config, color='blue')
            
            if comparison_data is not None:
                comp_lat = self.channel_series(comparison_data, 'gps_lat_acc')
                comp_lon = self.channel_series(comparison_data, 'gps_lon_acc')
                if comp_lat is not None and comp_lon is not None:
                    ax.scatter(comp_lat, comp_lon, alpha=0.5, s=20, label=comparison_config, color='red')
            
            ax.set_xlabel('GPS LatAcc (g)', fontsize=12)
            ax.set_ylabel('GPS LonAcc (g)', fontsize=12)
            ax.set_title(f'GG Diagram - {label}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_aspect('equal')
            
            # Add reference circles
            for radius in [0.5, 1.0, 1.5, 2.0]:
                circle = plt.Circle((0, 0), radius, fill=False, linestyle='--', 
                                   color='gray', alpha=0.3, linewidth=1)
                ax.add_patch(circle)
        
        filename = f'gg_diagram_{label}.png'
        plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
        return filename
    
    def _plot_speed_vs_yaw(self, label, data, comparison_data=None, primary_config='Primary', comparison_config='Comparison'):
        """Plot GPS Speed vs Yaw Rate"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        gps_speed = self.channel_series(data, 'gps_speed')
        yaw_rate = self.channel_series(data, 'yaw_rate')
        if gps_speed is not None and yaw_rate is not None:
            ax.scatter(yaw_rate, gps_speed, alpha=0.6, s=25, label=primary_config, color='blue')
            
            if comparison_data is not None:
                comp_speed = self.channel_series(comparison_data, 'gps_speed')
                comp_yaw = self.channel_series(comparison_data, 'yaw_rate')
                if comp_speed is not None and comp_yaw is not None:
                    ax.scatter(comp_yaw, comp_speed, alpha=0.6, s=25, label=comparison_config, color='red')
            
            ax.set_xlabel('Yaw Rate (deg/s)', fontsize=12)
            ax.set_ylabel('GPS Speed (km/h)', fontsize=12)
            ax.set_title(f'GPS Speed vs Yaw Rate - {label}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        filename = f'speed_vs_yaw_{label}.png'
        plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
        return filename
    
    def _plot_latacc_vs_yaw(self, label, data, comparison_data=None, primary_config='Primary', comparison_config='Comparison'):
        """Plot GPS LatAcc vs Yaw Rate"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        lat_acc = self.channel_series(data, 'gps_lat_acc')
        yaw_rate = self.channel_series(data, 'yaw_rate')
        if lat_acc is not None and yaw_rate is not None:
            ax.scatter(yaw_rate, lat_acc, alpha=0.6, s=25, label=primary_config, color='green')
            
            if comparison_data is not None:
                comp_lat = self.channel_series(comparison_data, 'gps_lat_acc')
                comp_yaw = self.channel_series(comparison_data, 'yaw_rate')
                if comp_lat is not None and comp_yaw is not None:
                    ax.scatter(comp_yaw, comp_lat, alpha=0.6, s=25, label=comparison_config, color='orange')
            
            ax.set_xlabel('Yaw Rate (deg/s)', fontsize=12)
            ax.set_ylabel('GPS LatAcc (g)', fontsize=12)
            ax.set_title(f'GPS LatAcc vs Yaw Rate - {label}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        filename = f'latacc_vs_yaw_{label}.png'
        plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
        return filename
    
    def _plot_speed_vs_wheelspeed(self, label, data, comparison_data=None, primary_config='Primary', comparison_config='Comparison'):
        """Plot GPS Speed vs RR and RL Wheel Speeds"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # RL Wheel Speed
        gps_speed = self.channel_series(data, 'gps_speed')
        rl_speed = self.channel_series(data, 'wheel_speed_rl')
        rr_speed = self.channel_series(data, 'wheel_speed_rr')
        if gps_speed is not None and rl_speed is not None:
            axes[0].scatter(gps_speed, rl_speed, alpha=0.6, s=25, 
                           label=primary_config, color='blue')
            comp_speed = self.channel_series(comparison_data, 'gps_speed') if comparison_data is not None else None
            comp_rl = self.channel_series(comparison_data, 'wheel_speed_rl') if comparison_data is not None else None
            if comp_speed is not None and comp_rl is not None:
                axes[0].scatter(comp_speed, comp_rl, alpha=0.6, s=25, label=comparison_config, color='red')
            axes[0].set_xlabel('GPS Speed (km/h)', fontsize=11)
            axes[0].set_ylabel('RL Wheel Speed (km/h)', fontsize=11)
            axes[0].set_title('GPS Speed vs RL Wheel Speed', fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
        
        # RR Wheel Speed
        if gps_speed is not None and rr_speed is not None:
            axes[1].scatter(gps_speed, rr_speed, alpha=0.6, s=25, 
                           label=primary_config, color='green')
            comp_speed = self.channel_series(comparison_data, 'gps_speed') if comparison_data is not None else None
            comp_rr = self.channel_series(comparison_data, 'wheel_speed_rr') if comparison_data is not None else None
            if comp_speed is not None and comp_rr is not None:
                axes[1].scatter(comp_speed, comp_rr, alpha=0.6, s=25, label=comparison_config, color='orange')
            axes[1].set_xlabel('GPS Speed (km/h)', fontsize=11)
            axes[1].set_ylabel('RR Wheel Speed (km/h)', fontsize=11)
            axes[1].set_title('GPS Speed vs RR Wheel Speed', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        
        plt.suptitle(f'GPS Speed vs Wheel Speeds - {label}', fontsize=14, fontweight='bold')
        filename = f'speed_vs_wheelspeed_{label}.png'
        plt.savefig(self.data_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
        return filename


def main():
    """Main execution"""
    import sys
    
    # Determine data directory
    data_dir = Path(__file__).parent
    
    # Ask user for analysis mode
    print("\n" + "="*60)
    print("FSAE Telemetry Analysis Tool")
    print("="*60)
    print("\nAnalysis Modes:")
    print("1. Comparison (Aero vs No-Aero)")
    print("2. Individual Analysis")
    
    mode_choice = input("\nSelect mode (1 or 2) [default: 1]: ").strip()
    
    if mode_choice == '2':
        mode = 'individual'
    else:
        mode = 'comparison'
    
    # Run analysis
    analyzer = FSAETelemetryAnalyzer(data_dir)
    analyzer.analyze(mode=mode)
    
    print("\n" + "="*60)
    print("Analysis complete! Check the workspace folder for plots.")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
