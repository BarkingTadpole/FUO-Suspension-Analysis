#!/usr/bin/env python3
"""Interactive Streamlit dashboard for FSAE telemetry analysis."""

from dataclasses import dataclass
from datetime import datetime
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid
import csv
import io

import pandas as pd
import streamlit as st

from dashboard_config import DASHBOARD_GRAPHS, DASHBOARD_TITLE
from math_channels import MathChannelEngine
from telemetry_analysis import FSAETelemetryAnalyzer


@dataclass
class UploadedRun:
    """A telemetry upload saved to a workspace with an assigned configuration."""

    original_name: str
    saved_path: Path
    config_label: str
    event_type: str = ""
    custom_event: str = ""
    driver: str = ""
    field: str = ""
    suspension_setup: str = ""
    sprocket_size: str = ""
    notes: str = ""
    file_id: str = ""

    @property
    def stem(self):
        return self.saved_path.stem

    @property
    def event_label(self):
        return self.custom_event if self.event_type == "Custom" and self.custom_event else self.event_type


class TelemetryDashboardApp:
    """Streamlit UI wrapper around FSAETelemetryAnalyzer."""

    def __init__(self):
        self.workspace_root = Path(tempfile.gettempdir()) / "fsae_telemetry_dashboard"
        self.sample_dir = Path(__file__).parent
        self.library_root = self.sample_dir / ".telemetry_uploads"
        self.library_files_root = self.library_root / "files"
        self.library_manifest_path = self.library_root / "manifest.json"
        self.sample_manifest_path = self.library_root / "sample_manifest.json"

    def run(self):
        st.set_page_config(page_title=DASHBOARD_TITLE, layout="wide")
        st.title(DASHBOARD_TITLE)
        st.caption(self._version_label())
        st.caption("Upload AiM CSV files, choose a mode, select graphs, and generate an interactive dashboard.")

        mode = st.sidebar.radio("Mode", ["comparison", "individual"], index=0)
        lap_threshold_percent = st.sidebar.slider("All-laps threshold (% of best lap)", 100, 150, 120, 5)
        comparison_lap_sets = self._comparison_lap_set_selector(mode)
        graph_ids = self._graph_selector()
        show_channel_summary = st.sidebar.checkbox("Show channel summary dashboard", value=True)
        use_samples = st.sidebar.checkbox("Use sample files in this folder", value=True)

        uploaded_runs = self._collect_runs(use_samples)
        self._render_run_table(uploaded_runs)
        self._track_replay_dashboard(uploaded_runs)
        if show_channel_summary and uploaded_runs:
            summary_df = self._build_channel_summary(uploaded_runs)
            self._display_channel_summary(summary_df)
        detected_channels = self._detect_channels(uploaded_runs)
        channel_units = self._detect_channel_units(uploaded_runs)
        math_channels = self._custom_math_channel_builder(detected_channels, channel_units)
        detected_channels = self._channels_with_math(detected_channels, math_channels)
        channel_units.update({channel["name"]: channel.get("unit", "") for channel in math_channels})
        custom_graphs = self._custom_graph_builder(detected_channels, channel_units)
        graph_filters = self._graph_filter_builder(graph_ids, custom_graphs, detected_channels)

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
                graph_filters,
                math_channels,
            )

    def _version_label(self):
        """Return a visible git version label for deployed Streamlit builds."""
        try:
            commit_subject = subprocess.check_output(
                ["git", "-C", str(self.sample_dir), "log", "-1", "--pretty=%s"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            commit_hash = subprocess.check_output(
                ["git", "-C", str(self.sample_dir), "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            dirty_status = subprocess.check_output(
                ["git", "-C", str(self.sample_dir), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            dirty_marker = " + local changes" if dirty_status else ""
            if commit_subject and commit_hash:
                return f"Version: {commit_hash} - {commit_subject}{dirty_marker}"
        except Exception:
            pass

        commit_message_path = self.sample_dir / ".git" / "COMMIT_EDITMSG"
        if commit_message_path.exists():
            commit_message = commit_message_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            commit_message = [line for line in commit_message if line and not line.startswith("#")]
            if commit_message:
                return f"Version: {commit_message[0]}"

        return "Version: local development build"

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

        manifest = self._load_file_manifest()
        if uploads:
            saved_count = self._save_uploaded_files(uploads, manifest)
            if saved_count:
                self._save_file_manifest(manifest)
                st.success(f"Saved {saved_count} uploaded file(s) to the persistent data library.")

        if use_samples:
            runs.extend(self._sample_file_library(workspace))

        runs.extend(self._persistent_file_library(manifest))

        return runs

    def _sample_file_library(self, workspace):
        manifest = self._load_sample_manifest()
        sample_files = [("265.csv", "No Aero"), ("266.csv", "Aero"), ("273.csv", "Aero")]
        for filename, config_label in sample_files:
            sample_path = self.sample_dir / filename
            if not sample_path.exists():
                continue
            sample_id = f"sample_{filename}"
            if sample_id not in manifest:
                metadata = self._metadata_from_aim_csv_path(sample_path)
                metadata["aero_config"] = metadata.get("aero_config") or config_label
                manifest[sample_id] = {
                    "id": sample_id,
                    "original_name": filename,
                    "stored_name": filename,
                    "hidden": False,
                    **metadata,
                }

        st.subheader("Sample File Library")
        hidden_samples = [entry for entry in manifest.values() if entry.get("hidden")]
        if hidden_samples and st.button("Restore Deleted Sample Files"):
            for entry in manifest.values():
                entry["hidden"] = False
            self._save_sample_manifest(manifest)
            st.rerun()

        runs = []
        visible_entries = [entry for entry in manifest.values() if not entry.get("hidden")]
        label_to_entry = {self._library_entry_label(entry): entry for entry in visible_entries}
        selected_labels = st.multiselect(
            "Sample files to include",
            options=list(label_to_entry.keys()),
            default=list(label_to_entry.keys()),
        )

        for entry in visible_entries:
            with st.expander(f"Sample Info - {entry.get('original_name')}", expanded=False):
                self._sample_entry_editor(entry, manifest)

            if self._library_entry_label(entry) not in selected_labels:
                continue
            source_path = self.sample_dir / entry["stored_name"]
            target_path = workspace / entry["stored_name"]
            shutil.copy2(source_path, target_path)
            runs.append(UploadedRun(
                entry["original_name"],
                target_path,
                entry.get("aero_config", "No Aero"),
                event_type=entry.get("event_type", ""),
                custom_event=entry.get("custom_event", ""),
                driver=entry.get("driver", ""),
                field=entry.get("field", ""),
                suspension_setup=entry.get("suspension_setup", ""),
                sprocket_size=entry.get("sprocket_size", ""),
                notes=entry.get("notes", ""),
                file_id=entry.get("id", ""),
            ))
        self._save_sample_manifest(manifest)
        return runs

    def _load_sample_manifest(self):
        if not self.sample_manifest_path.exists():
            return {}
        try:
            return json.loads(self.sample_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save_sample_manifest(self, manifest):
        self.library_root.mkdir(parents=True, exist_ok=True)
        self.sample_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _sample_entry_editor(self, entry, manifest):
        self._metadata_entry_editor(entry, lambda: self._save_sample_manifest(manifest))
        columns = st.columns([1, 4])
        if columns[0].button("Delete Sample", key=f"delete_{entry['id']}"):
            entry["hidden"] = True
            self._save_sample_manifest(manifest)
            st.rerun()

    def _load_file_manifest(self):
        if not self.library_manifest_path.exists():
            return {"files": []}
        try:
            return json.loads(self.library_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            st.warning("Upload library manifest is invalid. Starting with an empty library.")
            return {"files": []}

    def _save_file_manifest(self, manifest):
        self.library_root.mkdir(parents=True, exist_ok=True)
        self.library_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _save_uploaded_files(self, uploads, manifest):
        self.library_files_root.mkdir(parents=True, exist_ok=True)
        saved_count = 0
        existing_hashes = {entry.get("content_hash") for entry in manifest.get("files", []) if entry.get("content_hash")}
        for upload in uploads:
            upload_bytes = bytes(upload.getbuffer())
            content_hash = hashlib.sha256(upload_bytes).hexdigest()
            if content_hash in existing_hashes:
                continue

            header_metadata = self._metadata_from_aim_csv_bytes(upload_bytes, upload.name)
            file_id = uuid.uuid4().hex[:12]
            safe_name = self._safe_filename(upload.name)
            stored_name = f"{file_id}_{safe_name}"
            stored_path = self.library_files_root / stored_name
            stored_path.write_bytes(upload_bytes)
            manifest.setdefault("files", []).append({
                "id": file_id,
                "original_name": upload.name,
                "stored_name": stored_name,
                "uploaded_at": datetime.now().isoformat(timespec="seconds"),
                "content_hash": content_hash,
                "size_bytes": len(upload_bytes),
                **header_metadata,
            })
            existing_hashes.add(content_hash)
            saved_count += 1
        return saved_count

    def _default_file_metadata(self, filename):
        return {
            "event_type": "Autocross",
            "custom_event": "",
            "driver": "",
            "field": "",
            "suspension_setup": "",
            "aero_config": "Aero" if "aero" in filename.lower() and "no" not in filename.lower() else "No Aero",
            "sprocket_size": "",
            "notes": "",
        }

    def _metadata_from_aim_csv_path(self, path):
        return self._metadata_from_aim_csv_bytes(path.read_bytes(), path.name)

    def _metadata_from_aim_csv_bytes(self, data, filename):
        metadata = self._default_file_metadata(filename)
        aim_header = self._read_aim_header_metadata(data)

        session = aim_header.get("Session", "")
        event_type, custom_event = self._event_from_session(session)
        metadata.update({
            "event_type": event_type,
            "custom_event": custom_event,
            "driver": aim_header.get("Racer", ""),
            "field": aim_header.get("Championship", ""),
            "notes": self._notes_from_aim_header(aim_header),
        })

        comment = aim_header.get("Comment", "")
        inferred_aero = self._aero_from_text(f"{filename} {session} {comment}")
        if inferred_aero:
            metadata["aero_config"] = inferred_aero

        suspension_setup = self._regex_first(comment, [
            r"suspension(?:\s+setup)?\s*[:=-]\s*([^,;\n\r]+)",
            r"setup\s*[:=-]\s*([^,;\n\r]+)",
        ])
        sprocket_size = self._regex_first(comment, [
            r"sprocket(?:\s+size)?\s*[:=-]\s*([0-9]{1,2}\s*/\s*[0-9]{1,2}|[0-9]{1,2}[-xX][0-9]{1,2}|[^,;\n\r]+)",
            r"gear(?:ing)?\s*[:=-]\s*([0-9]{1,2}\s*/\s*[0-9]{1,2}|[0-9]{1,2}[-xX][0-9]{1,2})",
        ])
        if suspension_setup:
            metadata["suspension_setup"] = suspension_setup
        if sprocket_size:
            metadata["sprocket_size"] = sprocket_size

        return metadata

    def _read_aim_header_metadata(self, data):
        text = data.decode("utf-8-sig", errors="ignore")
        rows = list(csv.reader(io.StringIO(text)))
        metadata = {}
        for row in rows[:25]:
            if len(row) > 10 and row[0].strip('"').strip() == "Time":
                break
            if len(row) >= 2:
                key = row[0].strip('"').strip()
                value = row[1].strip('"').strip()
                if key:
                    metadata[key] = value
        return metadata

    def _event_from_session(self, session):
        normalized = self._normalize_text(session)
        event_map = [
            ("Autocross", ["autocross", "auto x", "autox", "auto-x"]),
            ("Accel", ["accel", "acceleration"]),
            ("Skidpad", ["skidpad", "skid pad"]),
            ("Amigo Track 1", ["amigo 1", "amigo track 1", "amigo one"]),
            ("Amigo Track 2", ["amigo 2", "amigo track 2", "amigo two"]),
        ]
        for event_type, patterns in event_map:
            if any(pattern in normalized for pattern in patterns):
                return event_type, ""
        if session:
            return "Custom", session
        return "Autocross", ""

    def _aero_from_text(self, value):
        normalized = self._normalize_text(value)
        if any(token in normalized for token in ["no aero", "no-aero", "without aero", "non aero", "nonaero"]):
            return "No Aero"
        if "aero" in normalized:
            return "Aero"
        return None

    def _notes_from_aim_header(self, aim_header):
        parts = []
        for key in ["Comment", "Vehicle", "Date", "Time", "Sample Rate", "Duration"]:
            value = aim_header.get(key)
            if value:
                parts.append(f"{key}: {value}")
        return "\n".join(parts)

    def _regex_first(self, value, patterns):
        for pattern in patterns:
            match = re.search(pattern, value or "", flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def _normalize_text(self, value):
        return re.sub(r"\s+", " ", str(value).lower()).strip()

    def _persistent_file_library(self, manifest):
        st.subheader("Persistent Data File Library")
        files = manifest.get("files", [])
        if not files:
            st.caption("Uploaded files will stay here until you delete them.")
            return []

        label_to_entry = {
            self._library_entry_label(entry): entry
            for entry in files
            if self._library_file_path(entry).exists()
        }
        selected_labels = st.multiselect(
            "Stored files to include",
            options=list(label_to_entry.keys()),
            default=list(label_to_entry.keys()),
        )

        runs = []
        for entry in list(files):
            file_path = self._library_file_path(entry)
            if not file_path.exists():
                continue

            with st.expander(f"File Info - {entry.get('original_name', file_path.name)}", expanded=False):
                self._library_entry_editor(entry, manifest)

            if self._library_entry_label(entry) in selected_labels:
                runs.append(self._run_from_library_entry(entry))

        return runs

    def _library_entry_label(self, entry):
        event_type = entry.get("custom_event") if entry.get("event_type") == "Custom" and entry.get("custom_event") else entry.get("event_type", "")
        driver = entry.get("driver", "")
        aero = entry.get("aero_config", "")
        parts = [entry.get("original_name", "uploaded.csv")]
        details = ", ".join(part for part in [event_type, driver, aero] if part)
        if details:
            parts.append(f"({details})")
        parts.append(f"[{entry.get('id', 'unknown')}]")
        return " ".join(parts)

    def _library_file_path(self, entry):
        return self.library_files_root / entry.get("stored_name", "")

    def _library_entry_editor(self, entry, manifest):
        self._metadata_entry_editor(entry, lambda: self._save_file_manifest(manifest))
        columns = st.columns([1, 1, 4])
        if columns[0].button("Auto Fill from CSV Header", key=f"autofill_{entry['id']}"):
            entry.update(self._metadata_from_aim_csv_path(self._library_file_path(entry)))
            self._save_file_manifest(manifest)
            st.rerun()
        if columns[1].button("Delete File", key=f"delete_file_{entry['id']}"):
            self._delete_library_entry(entry, manifest)
            st.rerun()

    def _metadata_entry_editor(self, entry, save_callback):
        event_options = ["Autocross", "Accel", "Skidpad", "Amigo Track 1", "Amigo Track 2", "Custom"]
        file_id = entry["id"]
        current_event = entry.get("event_type", "Autocross")
        event_index = event_options.index(current_event) if current_event in event_options else 0

        columns = st.columns(3)
        entry["event_type"] = columns[0].selectbox("Event", event_options, index=event_index, key=f"event_{file_id}")
        entry["aero_config"] = columns[1].selectbox(
            "Aero",
            ["No Aero", "Aero"],
            index=1 if entry.get("aero_config") == "Aero" else 0,
            key=f"aero_{file_id}",
        )
        entry["driver"] = columns[2].text_input("Driver", value=entry.get("driver", ""), key=f"driver_{file_id}")

        if entry["event_type"] == "Custom":
            entry["custom_event"] = st.text_input("Custom event name", value=entry.get("custom_event", ""), key=f"custom_event_{file_id}")
        else:
            entry["custom_event"] = ""

        columns = st.columns(3)
        entry["field"] = columns[0].text_input("Field", value=entry.get("field", ""), key=f"field_{file_id}")
        entry["suspension_setup"] = columns[1].text_input("Suspension setup", value=entry.get("suspension_setup", ""), key=f"suspension_{file_id}")
        entry["sprocket_size"] = columns[2].text_input("Sprocket size", value=entry.get("sprocket_size", ""), key=f"sprocket_{file_id}")
        entry["notes"] = st.text_area("Notes", value=entry.get("notes", ""), key=f"notes_{file_id}", height=80)

        columns = st.columns([1, 1, 1, 3])
        if columns[0].button("Save Info", key=f"save_info_{file_id}"):
            save_callback()
            st.success("Saved file info.")

    def _delete_library_entry(self, entry, manifest):
        file_path = self._library_file_path(entry)
        if file_path.exists():
            file_path.unlink()
        manifest["files"] = [item for item in manifest.get("files", []) if item.get("id") != entry.get("id")]
        self._save_file_manifest(manifest)

    def _run_from_library_entry(self, entry):
        return UploadedRun(
            original_name=entry.get("original_name", entry.get("stored_name", "uploaded.csv")),
            saved_path=self._library_file_path(entry),
            config_label=entry.get("aero_config", "No Aero"),
            event_type=entry.get("event_type", ""),
            custom_event=entry.get("custom_event", ""),
            driver=entry.get("driver", ""),
            field=entry.get("field", ""),
            suspension_setup=entry.get("suspension_setup", ""),
            sprocket_size=entry.get("sprocket_size", ""),
            notes=entry.get("notes", ""),
            file_id=entry.get("id", ""),
        )

    def _prepare_workspace(self):
        workspace = self.workspace_root
        workspace.mkdir(parents=True, exist_ok=True)
        for path in workspace.glob("*"):
            if path.is_file():
                path.unlink()
        return workspace

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
                    "Event": run.event_label,
                    "Driver": run.driver,
                    "Field": run.field,
                    "Suspension Setup": run.suspension_setup,
                    "Configuration": run.config_label,
                    "Sprocket Size": run.sprocket_size,
                }
                for run in runs
            ],
            width="stretch",
        )

    def _track_replay_dashboard(self, runs):
        st.subheader("Track Replay & Lap Review")
        if not runs:
            st.caption("Select files to view lap times, overlays, and moving track dots.")
            return

        with st.expander("Track maps, laps, times, overlays, and replay", expanded=False):
            lap_records = self._build_lap_records(runs)
            if not lap_records:
                st.info("No valid lap segments with GPS latitude/longitude were found.")
                return

            summary_df = pd.DataFrame([
                {
                    "Run": record["run"].original_name,
                    "Configuration": record["run"].config_label,
                    "Event": record["run"].event_label,
                    "Driver": record["run"].driver,
                    "Lap #": record["lap_number"],
                    "Lap Time (s)": round(record["duration"], 3),
                    "Distance (m)": round(record.get("distance") or 0, 1),
                }
                for record in lap_records
            ])
            st.dataframe(summary_df, width="stretch", height=280)

            best_by_run = summary_df.loc[summary_df.groupby("Run")["Lap Time (s)"].idxmin()].reset_index(drop=True)
            st.markdown("Best laps by run")
            st.dataframe(best_by_run, width="stretch")

            label_to_record = {self._lap_record_label(record): record for record in lap_records}
            default_labels = [self._lap_record_label(record) for record in lap_records if record["is_best"]]
            selected_labels = st.multiselect(
                "Laps to overlay/replay",
                options=list(label_to_record.keys()),
                default=default_labels[:6],
            )
            selected_records = [label_to_record[label] for label in selected_labels]
            if not selected_records:
                st.caption("Select one or more laps to draw the overlay and replay dots.")
                return

            replay_html = self._track_replay_html(selected_records)
            encoded_html = base64.b64encode(replay_html.encode("utf-8")).decode("ascii")
            st.iframe(f"data:text/html;base64,{encoded_html}", height=720, width="stretch")

    def _build_lap_records(self, runs):
        analyzer = FSAETelemetryAnalyzer(self.workspace_root, files=[])
        records = []
        for run in runs:
            try:
                df = analyzer.load_csv(run.saved_path)
            except Exception:
                continue
            if not analyzer.has_channels(df, ["gps_latitude", "gps_longitude"]):
                continue
            segments = analyzer.get_valid_lap_segments(df)
            if not segments:
                continue
            best_duration = min(segment["duration"] for segment in segments)
            for segment in segments:
                lap_data = analyzer._extract_segment_data(df, segment)
                gps_data = lap_data[[analyzer.resolve_channel(lap_data, "gps_latitude"), analyzer.resolve_channel(lap_data, "gps_longitude")]].dropna()
                if len(gps_data) < 2:
                    continue
                records.append({
                    "run": run,
                    "lap_data": lap_data,
                    "gps_data": gps_data,
                    "lap_number": segment["index"] + 1,
                    "duration": segment["duration"],
                    "distance": segment.get("distance"),
                    "is_best": segment["duration"] == best_duration,
                })
        return records

    def _lap_record_label(self, record):
        run = record["run"]
        best = " BEST" if record["is_best"] else ""
        return f"{run.original_name} lap {record['lap_number']} - {record['duration']:.3f}s - {run.config_label}{best}"

    def _track_replay_html(self, records):
        analyzer = FSAETelemetryAnalyzer(self.workspace_root, files=[])
        origin_lat = pd.concat([analyzer.channel_series(record["gps_data"], "gps_latitude") for record in records], ignore_index=True).mean()
        origin_lon = pd.concat([analyzer.channel_series(record["gps_data"], "gps_longitude") for record in records], ignore_index=True).mean()
        colors = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2", "#be123c", "#4b5563"]
        laps = []
        all_x = []
        all_y = []

        for idx, record in enumerate(records):
            x, y = analyzer._latlon_to_local_xy(record["gps_data"], origin_lat, origin_lon)
            points = [{"x": float(px), "y": float(py)} for px, py in zip(x, y)]
            all_x.extend([point["x"] for point in points])
            all_y.extend([point["y"] for point in points])
            laps.append({
                "label": self._lap_record_label(record),
                "color": colors[idx % len(colors)],
                "points": self._resample_points(points, 180),
                "path": points,
            })

        if not all_x or not all_y:
            return "<p>No GPS points available.</p>"

        bounds = {
            "minX": min(all_x),
            "maxX": max(all_x),
            "minY": min(all_y),
            "maxY": max(all_y),
        }
        payload = json.dumps({"laps": laps, "bounds": bounds})
        return f"""
<div style="font-family:Arial,sans-serif;">
  <div style="display:flex;gap:16px;align-items:center;margin-bottom:8px;">
    <button id="playPause" style="padding:6px 12px;">Play</button>
    <input id="frameSlider" type="range" min="0" max="179" value="0" style="width:360px;">
    <span id="frameText">Frame 0</span>
  </div>
  <svg id="trackSvg" viewBox="0 0 900 620" width="100%" height="620" style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;"></svg>
  <div id="legend" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:6px;margin-top:10px;"></div>
</div>
<script>
const data = {payload};
const svg = document.getElementById("trackSvg");
const slider = document.getElementById("frameSlider");
const frameText = document.getElementById("frameText");
const playPause = document.getElementById("playPause");
const legend = document.getElementById("legend");
const pad = 40, width = 900, height = 620;
const sx = x => pad + ((x - data.bounds.minX) / Math.max(1, data.bounds.maxX - data.bounds.minX)) * (width - 2 * pad);
const sy = y => height - pad - ((y - data.bounds.minY) / Math.max(1, data.bounds.maxY - data.bounds.minY)) * (height - 2 * pad);
const dots = [];
data.laps.forEach((lap, i) => {{
  const pathData = lap.path.map((p, idx) => `${{idx === 0 ? "M" : "L"}} ${{sx(p.x)}} ${{sy(p.y)}}`).join(" ");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", pathData);
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", lap.color);
  path.setAttribute("stroke-width", "2");
  path.setAttribute("opacity", "0.65");
  svg.appendChild(path);
  const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  dot.setAttribute("r", "7");
  dot.setAttribute("fill", lap.color);
  dot.setAttribute("stroke", "white");
  dot.setAttribute("stroke-width", "2");
  svg.appendChild(dot);
  dots.push(dot);
  const item = document.createElement("div");
  item.innerHTML = `<span style="display:inline-block;width:12px;height:12px;background:${{lap.color}};margin-right:6px;"></span>${{lap.label}}`;
  legend.appendChild(item);
}});
function update(frame) {{
  data.laps.forEach((lap, i) => {{
    const p = lap.points[Math.min(frame, lap.points.length - 1)];
    dots[i].setAttribute("cx", sx(p.x));
    dots[i].setAttribute("cy", sy(p.y));
  }});
  frameText.textContent = `Frame ${{frame}}`;
}}
let playing = false;
let timer = null;
playPause.onclick = () => {{
  playing = !playing;
  playPause.textContent = playing ? "Pause" : "Play";
  if (playing) {{
    timer = setInterval(() => {{
      slider.value = (Number(slider.value) + 1) % 180;
      update(Number(slider.value));
    }}, 50);
  }} else {{
    clearInterval(timer);
  }}
}};
slider.oninput = () => update(Number(slider.value));
update(0);
</script>
"""

    def _resample_points(self, points, count):
        if len(points) <= 1:
            return points
        target = [i * (len(points) - 1) / (count - 1) for i in range(count)]
        resampled = []
        for value in target:
            lower = int(value)
            upper = min(lower + 1, len(points) - 1)
            frac = value - lower
            resampled.append({
                "x": points[lower]["x"] * (1 - frac) + points[upper]["x"] * frac,
                "y": points[lower]["y"] * (1 - frac) + points[upper]["y"] * frac,
            })
        return resampled

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

    def _detect_channel_units(self, runs):
        """Return the first detected unit label for each CSV channel."""
        units = {}
        for run in runs:
            for channel, unit in self._read_channel_units(run.saved_path).items():
                units.setdefault(channel, unit)
        return units

    def _read_channel_units(self, filepath):
        import csv

        with open(filepath, "r") as f:
            rows = list(csv.reader(f))

        for idx, row in enumerate(rows):
            headers = [column.strip('"').strip() for column in row]
            if len(headers) > 10 and headers[0] == "Time" and idx + 1 < len(rows):
                unit_row = [unit.strip('"').strip() for unit in rows[idx + 1]]
                return {
                    header: unit
                    for header, unit in zip(headers, unit_row)
                    if header
                }
        return {}

    def _channels_with_math(self, detected_channels, math_channels):
        channels = list(detected_channels)
        for channel in math_channels:
            if channel["name"] not in channels:
                channels.append(channel["name"])
        return channels

    def _build_channel_summary(self, runs):
        """Build max/min/average stats for every numeric channel in each selected CSV."""
        analyzer = FSAETelemetryAnalyzer(self.workspace_root, files=[])
        rows = []

        for run in runs:
            try:
                df = analyzer.load_csv(run.saved_path)
            except Exception as exc:
                rows.append({
                    "File": run.original_name,
                    "Configuration": run.config_label,
                    "Channel": "CSV load error",
                    "Min": None,
                    "Max": None,
                    "Average": None,
                    "Samples": 0,
                    "Note": str(exc),
                })
                continue

            for column in df.columns:
                if df[column].dtype.kind not in "biufc":
                    continue

                series = df[column].dropna()
                if series.empty:
                    continue

                rows.append({
                    "File": run.original_name,
                    "Configuration": run.config_label,
                    "Event": run.event_label,
                    "Driver": run.driver,
                    "Field": run.field,
                    "Suspension Setup": run.suspension_setup,
                    "Sprocket Size": run.sprocket_size,
                    "Channel": column,
                    "Min": float(series.min()),
                    "Max": float(series.max()),
                    "Average": float(series.mean()),
                    "Samples": int(series.count()),
                    "Note": "",
                })

        return pd.DataFrame(rows, columns=[
            "File",
            "Configuration",
            "Event",
            "Driver",
            "Field",
            "Suspension Setup",
            "Sprocket Size",
            "Channel",
            "Min",
            "Max",
            "Average",
            "Samples",
            "Note",
        ])

    def _display_channel_summary(self, summary_df):
        st.subheader("Channel Summary Dashboard")

        if summary_df.empty:
            st.info("No numeric channels found in the selected CSV files.")
            return

        dashboard_path = self._write_channel_summary_dashboard(summary_df)
        st.markdown(f"HTML channel summary saved at `{dashboard_path}`")

        display_df = summary_df.copy()
        for column in ["Min", "Max", "Average"]:
            display_df[column] = display_df[column].round(4)

        st.dataframe(display_df, width="stretch", height=360)
        st.download_button(
            "Download channel summary CSV",
            data=summary_df.to_csv(index=False),
            file_name="channel_summary.csv",
            mime="text/csv",
        )

    def _write_channel_summary_dashboard(self, summary_df):
        output_path = self.workspace_root / "dashboard_channel_summary.html"

        html_df = summary_df.copy()
        for column in ["Min", "Max", "Average"]:
            html_df[column] = html_df[column].round(4)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{DASHBOARD_TITLE} - Channel Summary</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 24px;
            color: #1f2933;
            background: #f7f9fb;
        }}
        h1 {{
            margin-bottom: 4px;
        }}
        .subtitle {{
            margin-top: 0;
            color: #52606d;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background: white;
            font-size: 13px;
        }}
        th, td {{
            border: 1px solid #d9e2ec;
            padding: 8px 10px;
            text-align: left;
        }}
        th {{
            background: #e4e7eb;
            position: sticky;
            top: 0;
        }}
        tr:nth-child(even) {{
            background: #f7f9fb;
        }}
    </style>
</head>
<body>
    <h1>{DASHBOARD_TITLE} - Channel Summary</h1>
    <p class="subtitle">Max, min, average, and sample count for every numeric CSV channel in the selected runs.</p>
    {html_df.to_html(index=False, escape=True)}
</body>
</html>
"""
        output_path.write_text(html, encoding="utf-8")
        return output_path

    def _custom_math_channel_builder(self, detected_channels, channel_units):
        """Build custom math channel definitions with calculator-style controls."""
        st.subheader("Custom Math Channels")
        if not detected_channels:
            st.caption("Select or upload CSV files to create math channels.")
            return []

        math_channels = []
        local_units = dict(channel_units)
        channel_count = st.number_input("Number of custom math channels", min_value=0, max_value=12, value=0, step=1)
        for idx in range(int(channel_count)):
            with st.expander(f"Math Channel {idx + 1}", expanded=True):
                available_channels = self._channels_with_math(detected_channels, math_channels)
                default_name = f"Math Channel {idx + 1}"
                name = st.text_input("Channel name", value=default_name, key=f"math_name_{idx}").strip()
                expression_key = f"math_expression_{idx}"
                if expression_key not in st.session_state:
                    st.session_state[expression_key] = ""

                insert_channel = st.selectbox("Insert channel", available_channels, key=f"math_insert_channel_{idx}")
                if st.button("Insert selected channel", key=f"math_insert_button_{idx}"):
                    self._append_math_token(expression_key, f"{{{insert_channel}}}")

                self._calculator_buttons(expression_key, idx)
                expression = st.text_area(
                    "Expression",
                    key=expression_key,
                    help="Use channels in braces, for example: {GPS Speed} * sin({GPS LatAcc})",
                ).strip()

                if not name or not expression:
                    st.caption("Enter a name and expression to enable this math channel.")
                    continue
                if name in available_channels:
                    st.warning(f"'{name}' already exists. Pick a unique math channel name.")
                    continue

                try:
                    unit = MathChannelEngine.infer_unit(expression, local_units)
                    missing_channels = [channel for channel in MathChannelEngine.channel_names(expression) if channel not in available_channels]
                    if missing_channels:
                        st.warning(f"Unknown input channel(s): {', '.join(missing_channels)}")
                        continue
                    st.caption(f"Auto unit: `{unit or 'unitless'}`")
                    math_channels.append({
                        "name": name,
                        "expression": expression,
                        "unit": unit,
                    })
                    local_units[name] = unit
                except Exception as exc:
                    st.warning(f"Math channel '{name}' is not valid yet: {exc}")

        return math_channels

    def _calculator_buttons(self, expression_key, idx):
        button_rows = [
            ["+", "-", "*", "/", "**", "(", ")"],
            ["sin(", "cos(", "tan(", "asin(", "acos(", "atan("],
            ["derivative(", "integral("],
            ["sqrt(", "abs(", "log(", "log10(", "exp(", "radians(", "degrees("],
            ["pi", "e", ".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        ]
        for row_idx, row in enumerate(button_rows):
            columns = st.columns(len(row))
            for col, token in zip(columns, row):
                if col.button(token, key=f"math_token_{idx}_{row_idx}_{token}"):
                    self._append_math_token(expression_key, token)

    def _append_math_token(self, expression_key, token):
        expression = st.session_state.get(expression_key, "")
        separator = "" if not expression or expression.endswith((" ", "(", "{", "+", "-", "*", "/", ".")) else " "
        st.session_state[expression_key] = f"{expression}{separator}{token}"

    def _custom_graph_builder(self, detected_channels, channel_units):
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
                    x_units = st.text_input("X units/label", value=self._channel_label(x_channel, channel_units), key=f"custom_x_units_{idx}")
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
                    x_units = st.text_input("X units/label", value=self._channel_label(x_channel, channel_units), key=f"custom_scatter_x_units_{idx}")
                    y_units = st.text_input("Y units/label", value=self._channel_label(y_channel, channel_units), key=f"custom_scatter_y_units_{idx}")
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

    def _channel_label(self, channel, channel_units):
        unit = channel_units.get(channel, "")
        return f"{channel} [{unit}]" if unit else channel

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

    def _graph_filter_builder(self, graph_ids, custom_graphs, detected_channels):
        """Build per-graph channel range filters from UI selections."""
        st.subheader("Graph Filters")
        if not detected_channels:
            st.caption("Select or upload CSV files to enable graph filters.")
            return {}

        graph_lookup = {graph["id"]: graph for graph in DASHBOARD_GRAPHS + custom_graphs}
        selected_graphs = [graph_lookup[graph_id] for graph_id in graph_ids if graph_id in graph_lookup]
        selected_graphs.extend(custom_graphs)
        if not selected_graphs:
            st.caption("Select at least one graph to configure filters.")
            return {}

        graph_filters = {}
        for graph in selected_graphs:
            label = f"{graph.get('category', 'General')} / {graph['name']}"
            with st.expander(f"Filters - {label}", expanded=False):
                enabled = st.checkbox("Enable filters for this graph", value=False, key=f"filter_enabled_{graph['id']}")
                if not enabled:
                    continue

                filter_count = st.number_input(
                    "Number of channel filters",
                    min_value=1,
                    max_value=8,
                    value=1,
                    step=1,
                    key=f"filter_count_{graph['id']}",
                )
                channel_options = self._graph_filter_channel_options(graph, detected_channels)
                filters = []
                for idx in range(int(filter_count)):
                    columns = st.columns([2, 1, 1])
                    channel = columns[0].selectbox(
                        "Channel",
                        channel_options,
                        key=f"filter_channel_{graph['id']}_{idx}",
                    )
                    min_value = columns[1].text_input("Min", value="", key=f"filter_min_{graph['id']}_{idx}")
                    max_value = columns[2].text_input("Max", value="", key=f"filter_max_{graph['id']}_{idx}")
                    parsed_filter = {
                        "channel": channel,
                        "min": self._parse_optional_float(min_value),
                        "max": self._parse_optional_float(max_value),
                    }
                    if parsed_filter["min"] is not None or parsed_filter["max"] is not None:
                        filters.append(parsed_filter)

                if filters:
                    graph_filters[graph["id"]] = filters
                else:
                    st.caption("Enter a min or max value to activate a filter.")
        return graph_filters

    def _graph_filter_channel_options(self, graph, detected_channels):
        graph_channels = []
        for key in ["x", "y", "speed_channel"]:
            if graph.get(key):
                graph_channels.append(graph[key])
        for channel in graph.get("channels", []):
            graph_channels.append(channel.get("id"))
        for panel in graph.get("panels", []):
            for key in ["x", "y", "channel"]:
                if panel.get(key):
                    graph_channels.append(panel[key])
        for corner in graph.get("corners", []):
            if corner.get("channel"):
                graph_channels.append(corner["channel"])

        options = []
        for channel in graph_channels + detected_channels:
            if channel and channel not in options:
                options.append(channel)
        return options

    def _parse_optional_float(self, value):
        value = str(value).strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            st.warning(f"Ignoring invalid filter value: {value}")
            return None

    def _has_comparison_groups(self, runs):
        labels = {run.config_label for run in runs}
        return "No Aero" in labels and "Aero" in labels

    def _generate_dashboard(self, runs, mode, graph_ids, lap_threshold, custom_graphs, comparison_lap_sets, graph_filters, math_channels):
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
            graph_filters=graph_filters,
            math_channels=math_channels,
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
                        column.image(str(image_path), caption=plot["title"], width="stretch")


if __name__ == "__main__":
    TelemetryDashboardApp().run()
