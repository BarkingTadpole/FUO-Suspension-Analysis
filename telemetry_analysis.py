"""
FSAE Telemetry Analysis Tool
Analyzes shock data, acceleration, and vehicle dynamics
Supports aero vs no-aero comparison and individual analysis
"""

import pandas as pd
import numpy as np
import warnings
import copy
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
        graph_filters=None,
    ):
        """Initialize the analyzer with data directory"""
        self.data_dir = Path(data_dir)
        self.files = [Path(file) for file in files] if files else list(self.data_dir.glob('*.csv'))
        self.graph_ids = graph_ids
        self.config_labels = config_labels or {}
        self.lap_threshold = lap_threshold
        self.extra_graphs = extra_graphs or []
        self.comparison_lap_sets = comparison_lap_sets or ['best', 'all']
        self.graph_filters = graph_filters or {}
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
        available_graphs = [self._graph_with_filters(graph) for graph in all_graphs if graph.get('enabled', True)]
        if not self.graph_ids:
            return available_graphs

        graph_ids = set(self.graph_ids)
        graphs = [self._graph_with_filters(graph) for graph in all_graphs if graph['id'] in graph_ids]
        missing = sorted(graph_ids - {graph['id'] for graph in all_graphs})
        if missing:
            raise ValueError(f"Unknown graph id(s): {', '.join(missing)}")
        return graphs

    def _graph_with_filters(self, graph):
        """Return a graph config copy with any UI-supplied filters attached."""
        graph_copy = copy.deepcopy(graph)
        filters = self.graph_filters.get(graph_copy['id'])
        if filters:
            graph_copy['filters'] = filters
        return graph_copy

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

def main():
    """Main execution"""
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
