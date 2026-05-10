#!/usr/bin/env python3
"""Interactive Streamlit dashboard for FSAE telemetry analysis."""

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import tempfile

import streamlit as st

from dashboard_config import DASHBOARD_GRAPHS, DASHBOARD_TITLE
from telemetry_analysis import FSAETelemetryAnalyzer


@dataclass
class UploadedRun:
    """A telemetry upload saved to a workspace with an assigned configuration."""

    original_name: str
    saved_path: Path
    config_label: str

    @property
    def stem(self):
        return self.saved_path.stem


class TelemetryDashboardApp:
    """Streamlit UI wrapper around FSAETelemetryAnalyzer."""

    def __init__(self):
        self.workspace_root = Path(tempfile.gettempdir()) / "fsae_telemetry_dashboard"
        self.sample_dir = Path(__file__).parent

    def run(self):
        st.set_page_config(page_title=DASHBOARD_TITLE, layout="wide")
        st.title(DASHBOARD_TITLE)
        st.caption("Upload AiM CSV files, choose a mode, select graphs, and generate an interactive dashboard.")

        mode = st.sidebar.radio("Mode", ["comparison", "individual"], index=0)
        lap_threshold_percent = st.sidebar.slider("All-laps threshold (% of best lap)", 100, 150, 120, 5)
        comparison_lap_sets = self._comparison_lap_set_selector(mode)
        graph_ids = self._graph_selector()
        use_samples = st.sidebar.checkbox("Use sample files in this folder", value=True)

        uploaded_runs = self._collect_runs(use_samples)
        self._render_run_table(uploaded_runs)
        detected_channels = self._detect_channels(uploaded_runs)
        custom_graphs = self._custom_graph_builder(detected_channels)

        if st.button("Generate Dashboard", type="primary"):
            if not uploaded_runs:
                st.error("Upload at least one CSV file or enable the sample files.")
                return
            if mode == "comparison" and not self._has_comparison_groups(uploaded_runs):
                st.error("Comparison mode needs at least one No Aero file and one Aero file.")
                return
            selected_graph_ids = graph_ids + [graph["id"] for graph in custom_graphs]
            self._generate_dashboard(
                uploaded_runs,
                mode,
                selected_graph_ids,
                lap_threshold_percent / 100,
                custom_graphs,
                comparison_lap_sets,
            )

    def _comparison_lap_set_selector(self, mode):
        if mode != "comparison":
            return ["best"]
        selected = st.sidebar.multiselect(
            "Comparison lap sets",
            options=["Best laps only", "All laps within threshold"],
            default=["Best laps only", "All laps within threshold"],
        )
        lap_sets = []
        if "Best laps only" in selected:
            lap_sets.append("best")
        if "All laps within threshold" in selected:
            lap_sets.append("all")
        return lap_sets or ["best"]

    def _graph_selector(self):
        graph_options = {
            f"{graph.get('category', 'General')} / {graph['name']} ({graph['id']})": graph['id']
            for graph in DASHBOARD_GRAPHS
        }
        default_options = [
            f"{graph.get('category', 'General')} / {graph['name']} ({graph['id']})"
            for graph in DASHBOARD_GRAPHS
            if graph.get("enabled", True)
        ]
        selected = st.sidebar.multiselect(
            "Graphs",
            options=list(graph_options.keys()),
            default=default_options,
        )
        return [graph_options[label] for label in selected]

    def _collect_runs(self, use_samples):
        uploads = st.file_uploader("Upload AiM CSV files", type=["csv"], accept_multiple_files=True)
        workspace = self._prepare_workspace()
        runs = []

        if use_samples:
            for filename, config_label in [("265.csv", "No Aero"), ("266.csv", "Aero"), ("273.csv", "Aero")]:
                sample_path = self.sample_dir / filename
                if sample_path.exists():
                    target_path = workspace / filename
                    shutil.copy2(sample_path, target_path)
                    runs.append(UploadedRun(filename, target_path, config_label))

        for upload in uploads:
            config_label = self._config_picker_for_upload(upload.name)
            target_path = workspace / self._safe_filename(upload.name)
            target_path.write_bytes(upload.getbuffer())
            runs.append(UploadedRun(upload.name, target_path, config_label))

        return runs

    def _prepare_workspace(self):
        workspace = self.workspace_root
        workspace.mkdir(parents=True, exist_ok=True)
        for path in workspace.glob("*"):
            if path.is_file():
                path.unlink()
        return workspace

    def _config_picker_for_upload(self, filename):
        default_index = 1 if "aero" in filename.lower() and "no" not in filename.lower() else 0
        return st.selectbox(
            f"Configuration for {filename}",
            ["No Aero", "Aero"],
            index=default_index,
            key=f"config_{filename}",
        )

    def _safe_filename(self, filename):
        clean_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename).strip("_")
        return clean_name or "uploaded.csv"

    def _render_run_table(self, runs):
        if not runs:
            st.info("No files selected yet.")
            return
        st.subheader("Selected Runs")
        st.dataframe(
            [
                {
                    "File": run.original_name,
                    "Saved As": run.saved_path.name,
                    "Configuration": run.config_label,
                }
                for run in runs
            ],
            use_container_width=True,
        )

    def _detect_channels(self, runs):
        """Autodetect available numeric channels from selected CSV files."""
        if not runs:
            return []
        analyzer = FSAETelemetryAnalyzer(self.workspace_root, files=[])
        channels = []
        seen = set()
        for run in runs:
            try:
                df = analyzer.load_csv(run.saved_path)
            except Exception:
                continue
            for column in df.columns:
                if column in seen:
                    continue
                if df[column].dtype.kind in "biufc":
                    channels.append(column)
                    seen.add(column)
        return channels

    def _custom_graph_builder(self, detected_channels):
        """Build custom graph definitions from UI selections."""
        st.subheader("Custom Graph Builder")
        if not detected_channels:
            st.caption("Select or upload CSV files to detect channels.")
            return []

        custom_graphs = []
        graph_count = st.number_input("Number of custom graphs", min_value=0, max_value=10, value=0, step=1)
        graph_type_options = {
            "Time Series": "timeseries",
            "Scatter Plot": "scatter",
            "Percent Histogram": "histogram_percent",
            "Track Map": "track_map",
        }

        for idx in range(int(graph_count)):
            with st.expander(f"Custom Graph {idx + 1}", expanded=True):
                name = st.text_input("Graph name", value=f"Custom Graph {idx + 1}", key=f"custom_name_{idx}")
                graph_type_label = st.selectbox("Graph type", list(graph_type_options.keys()), key=f"custom_type_{idx}")
                graph_kind = graph_type_options[graph_type_label]
                graph_id = f"custom_{idx + 1}_{self._safe_filename(name).lower()}"
                graph = {
                    "id": graph_id,
                    "name": name,
                    "category": "Custom Graphs",
                    "kind": graph_kind,
                    "enabled": True,
                }

                if graph_kind == "timeseries":
                    x_channel = st.selectbox("X channel", detected_channels, index=self._default_channel_index(detected_channels, "Time"), key=f"custom_x_{idx}")
                    y_channels = st.multiselect("Y channel(s)", detected_channels, key=f"custom_y_multi_{idx}")
                    x_units = st.text_input("X units/label", value=x_channel, key=f"custom_x_units_{idx}")
                    y_units = st.text_input("Y units/label", value="Value", key=f"custom_y_units_{idx}")
                    graph.update({
                        "x": x_channel,
                        "x_label": x_units,
                        "y_label": y_units,
                        "channels": [{"id": channel, "label": channel} for channel in y_channels],
                    })

                elif graph_kind == "scatter":
                    x_channel = st.selectbox("X channel", detected_channels, key=f"custom_scatter_x_{idx}")
                    y_channel = st.selectbox("Y channel", detected_channels, key=f"custom_scatter_y_{idx}")
                    x_units = st.text_input("X units/label", value=x_channel, key=f"custom_scatter_x_units_{idx}")
                    y_units = st.text_input("Y units/label", value=y_channel, key=f"custom_scatter_y_units_{idx}")
                    graph.update({
                        "x": x_channel,
                        "y": y_channel,
                        "x_label": x_units,
                        "y_label": y_units,
                    })

                elif graph_kind == "histogram_percent":
                    channels = st.multiselect("Histogram channel(s)", detected_channels, key=f"custom_hist_channels_{idx}")
                    bins = st.number_input("Bins", min_value=4, max_value=80, value=12, step=1, key=f"custom_hist_bins_{idx}")
                    graph.update({
                        "channels": [{"id": channel, "label": channel} for channel in channels],
                        "bins": int(bins),
                    })

                elif graph_kind == "track_map":
                    st.caption("Track maps use GPS latitude/longitude channels through the channel resolver.")

                if self._custom_graph_is_valid(graph):
                    custom_graphs.append(graph)
                else:
                    st.caption("Pick the required channels for this graph to enable it.")
        return custom_graphs

    def _default_channel_index(self, channels, preferred):
        for idx, channel in enumerate(channels):
            if preferred.lower() in channel.lower():
                return idx
        return 0

    def _custom_graph_is_valid(self, graph):
        kind = graph["kind"]
        if kind == "timeseries":
            return bool(graph.get("x") and graph.get("channels"))
        if kind == "scatter":
            return bool(graph.get("x") and graph.get("y"))
        if kind == "histogram_percent":
            return bool(graph.get("channels"))
        if kind == "track_map":
            return True
        return False

    def _has_comparison_groups(self, runs):
        labels = {run.config_label for run in runs}
        return "No Aero" in labels and "Aero" in labels

    def _generate_dashboard(self, runs, mode, graph_ids, lap_threshold, custom_graphs, comparison_lap_sets):
        files = [run.saved_path for run in runs]
        config_labels = {run.stem: run.config_label for run in runs}
        analyzer = FSAETelemetryAnalyzer(
            self.workspace_root,
            files=files,
            graph_ids=graph_ids,
            config_labels=config_labels,
            lap_threshold=lap_threshold,
            extra_graphs=custom_graphs,
            comparison_lap_sets=comparison_lap_sets,
        )

        with st.spinner("Generating plots..."):
            analyzer.analyze(mode=mode)

        self._display_results(analyzer, mode)

    def _display_results(self, analyzer, mode):
        st.success(f"Generated {len(analyzer.generated_plots)} plots.")
        dashboard_path = self.workspace_root / f"dashboard_{mode}.html"
        if dashboard_path.exists():
            st.markdown(f"HTML dashboard saved at `{dashboard_path}`")

        sections = []
        for plot in analyzer.generated_plots:
            if plot["section"] not in sections:
                sections.append(plot["section"])

        for section in sections:
            st.header(section)
            section_plots = [plot for plot in analyzer.generated_plots if plot["section"] == section]
            for row_start in range(0, len(section_plots), 2):
                columns = st.columns(2)
                for column, plot in zip(columns, section_plots[row_start:row_start + 2]):
                    image_path = self.workspace_root / plot["filename"]
                    if image_path.exists():
                        column.image(str(image_path), caption=plot["title"], use_container_width=True)


if __name__ == "__main__":
    TelemetryDashboardApp().run()
