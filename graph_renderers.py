"""Reusable graph renderer classes for configurable telemetry dashboards."""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class GraphRenderer(ABC):
    """Base class for configurable graph renderers."""

    def __init__(self, analyzer, graph):
        self.analyzer = analyzer
        self.graph = graph

    @abstractmethod
    def render_comparison(self, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        """Render comparison-mode graph and return output filename."""

    @abstractmethod
    def render_individual(self, label, data, config_label):
        """Render individual-mode graph and return output filename."""

    def _filename(self, suffix):
        return f"{self.analyzer._slugify(self.graph['id'])}_{suffix}.png"

    def _save(self, fig, filename):
        fig.savefig(self.analyzer.data_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {filename}")
        return filename

    def _combine_laps(self, laps):
        return self.analyzer._combine_laps(laps)

    def _filtered_laps(self, laps):
        return [self._filter_data(lap) for lap in laps]

    def _filter_data(self, data):
        """Apply configured numeric channel filters before plotting."""
        filters = [item for item in self.graph.get("filters", []) if item.get("channel")]
        if data is None or data.empty or not filters:
            return data

        mask = pd.Series(True, index=data.index)
        for item in filters:
            series = self.analyzer.channel_series(data, item["channel"])
            if series is None:
                continue
            if item.get("min") is not None:
                mask &= series >= item["min"]
            if item.get("max") is not None:
                mask &= series <= item["max"]
        return data.loc[mask].copy()


class TimeSeriesGraph(GraphRenderer):
    """One axis with one or more y channels plotted against an x channel."""

    def render_comparison(self, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        fig, ax = plt.subplots(figsize=tuple(self.graph.get("figsize", (12, 6))))
        plotted = False
        x_channel = self.graph.get("x", "time")
        for laps, label_text, color in [
            (no_aero_laps, no_aero_label, "#1f77b4"),
            (aero_laps, aero_label, "#d62728"),
        ]:
            first_line = True
            for lap in self._filtered_laps(laps):
                x = self.analyzer.channel_series(lap, x_channel)
                if x is None:
                    continue
                for channel in self.graph.get("channels", []):
                    y = self.analyzer.channel_series(lap, channel["id"])
                    if y is None:
                        continue
                    line_label = f"{label_text} {channel.get('label', channel['id'])}" if first_line else None
                    ax.plot(x, y, linewidth=1.0, alpha=0.45, color=color, label=line_label)
                    plotted = True
                first_line = False
        if not plotted:
            print(f"  Skipped {self.graph['name']}: no configured channels found")
            plt.close(fig)
            return None
        ax.set_xlabel(self.graph.get("x_label", x_channel))
        ax.set_ylabel(self.graph.get("y_label", self.graph["name"]))
        ax.set_title(f"{self.graph['name']} - {plot_context}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return self._save(fig, self._filename(f"{filename_suffix}_comparison"))

    def render_individual(self, label, data, config_label):
        data = self._filter_data(data)
        fig, ax = plt.subplots(figsize=tuple(self.graph.get("figsize", (12, 6))))
        x_channel = self.graph.get("x", "time")
        x = self.analyzer.channel_series(data, x_channel)
        if x is None:
            print(f"  Skipped {self.graph['name']} for {label}: missing x channel {x_channel}")
            plt.close(fig)
            return None
        plotted = False
        for channel in self.graph.get("channels", []):
            y = self.analyzer.channel_series(data, channel["id"])
            if y is None:
                continue
            ax.plot(x, y, linewidth=1.2, alpha=0.85, label=channel.get("label", channel["id"]))
            plotted = True
        if not plotted:
            print(f"  Skipped {self.graph['name']} for {label}: no configured channels found")
            plt.close(fig)
            return None
        ax.set_xlabel(self.graph.get("x_label", x_channel))
        ax.set_ylabel(self.graph.get("y_label", self.graph["name"]))
        ax.set_title(f"{self.graph['name']} - {label}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return self._save(fig, self._filename(label))


class TimeSeriesGridGraph(GraphRenderer):
    """Multiple stacked time-series axes from config panels."""

    def render_comparison(self, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        panels = self.graph.get("panels", [])
        fig, axes = plt.subplots(len(panels), 1, figsize=tuple(self.graph.get("figsize", (14, 10))))
        axes = np.atleast_1d(axes)
        x_channel = self.graph.get("x", "time")
        for ax, panel in zip(axes, panels):
            for laps, label_text, color in [
                (no_aero_laps, no_aero_label, panel.get("no_aero_color", "#1f77b4")),
                (aero_laps, aero_label, panel.get("aero_color", "#d62728")),
            ]:
                for idx, lap in enumerate(self._filtered_laps(laps)):
                    x = self.analyzer.channel_series(lap, x_channel)
                    y = self.analyzer.channel_series(lap, panel["channel"])
                    if x is None or y is None:
                        continue
                    ax.plot(x, y, label=label_text if idx == 0 else None, linewidth=1.0, alpha=0.45, color=color)
            ax.set_ylabel(panel.get("label", panel["channel"]))
            ax.grid(True, alpha=0.3)
            ax.legend()
        axes[-1].set_xlabel(self.graph.get("x_label", x_channel))
        fig.suptitle(f"{self.graph['name']} - {plot_context}", fontsize=14, fontweight="bold")
        return self._save(fig, self._filename(f"{filename_suffix}_comparison"))

    def render_individual(self, label, data, config_label):
        data = self._filter_data(data)
        panels = self.graph.get("panels", [])
        fig, axes = plt.subplots(len(panels), 1, figsize=tuple(self.graph.get("figsize", (14, 10))))
        axes = np.atleast_1d(axes)
        x = self.analyzer.channel_series(data, self.graph.get("x", "time"))
        if x is None:
            print(f"  Skipped {self.graph['name']} for {label}: missing time channel")
            plt.close(fig)
            return None
        for ax, panel in zip(axes, panels):
            y = self.analyzer.channel_series(data, panel["channel"])
            if y is None:
                ax.set_visible(False)
                continue
            ax.plot(x, y, label=panel.get("label", panel["channel"]), linewidth=1.2, alpha=0.85)
            ax.set_ylabel(panel.get("label", panel["channel"]))
            ax.grid(True, alpha=0.3)
            ax.legend()
        axes[-1].set_xlabel(self.graph.get("x_label", "time"))
        fig.suptitle(f"{self.graph['name']} - {label}", fontsize=14, fontweight="bold")
        return self._save(fig, self._filename(label))


class ScatterGraph(GraphRenderer):
    """Single x/y scatter plot."""

    def render_comparison(self, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        no_aero_data = self._filter_data(self._combine_laps(no_aero_laps))
        aero_data = self._filter_data(self._combine_laps(aero_laps))
        fig, ax = plt.subplots(figsize=tuple(self.graph.get("figsize", (10, 7))))
        plotted = False
        for data, label_text, color in [
            (no_aero_data, no_aero_label, "#1f77b4"),
            (aero_data, aero_label, "#d62728"),
        ]:
            x = self.analyzer.channel_series(data, self.graph["x"])
            y = self.analyzer.channel_series(data, self.graph["y"])
            if x is None or y is None:
                continue
            ax.scatter(x, y, alpha=0.4, s=self.graph.get("size", 22), label=label_text, color=color)
            plotted = True
        if not plotted:
            print(f"  Skipped {self.graph['name']}: missing x/y channel")
            plt.close(fig)
            return None
        ax.set_xlabel(self.graph.get("x_label", self.graph["x"]))
        ax.set_ylabel(self.graph.get("y_label", self.graph["y"]))
        ax.set_title(f"{self.graph['name']} - {plot_context}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if self.graph.get("equal_aspect"):
            ax.set_aspect("equal")
            for radius in self.graph.get("reference_circles", []):
                ax.add_patch(plt.Circle((0, 0), radius, fill=False, linestyle="--", color="gray", alpha=0.3, linewidth=1))
        return self._save(fig, self._filename(f"{filename_suffix}_comparison"))

    def render_individual(self, label, data, config_label):
        data = self._filter_data(data)
        x = self.analyzer.channel_series(data, self.graph["x"])
        y = self.analyzer.channel_series(data, self.graph["y"])
        if x is None or y is None:
            print(f"  Skipped {self.graph['name']} for {label}: missing x/y channel")
            return None
        fig, ax = plt.subplots(figsize=tuple(self.graph.get("figsize", (10, 7))))
        ax.scatter(x, y, alpha=0.5, s=self.graph.get("size", 22), label=config_label)
        ax.set_xlabel(self.graph.get("x_label", self.graph["x"]))
        ax.set_ylabel(self.graph.get("y_label", self.graph["y"]))
        ax.set_title(f"{self.graph['name']} - {label}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if self.graph.get("equal_aspect"):
            ax.set_aspect("equal")
            for radius in self.graph.get("reference_circles", []):
                ax.add_patch(plt.Circle((0, 0), radius, fill=False, linestyle="--", color="gray", alpha=0.3, linewidth=1))
        return self._save(fig, self._filename(label))


class ScatterGridGraph(GraphRenderer):
    """Multiple scatter plots in one figure."""

    def render_comparison(self, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        panels = self.graph.get("panels", [])
        fig, axes = plt.subplots(1, len(panels), figsize=tuple(self.graph.get("figsize", (14, 6))))
        axes = np.atleast_1d(axes)
        no_aero_data = self._filter_data(self._combine_laps(no_aero_laps))
        aero_data = self._filter_data(self._combine_laps(aero_laps))
        for ax, panel in zip(axes, panels):
            for data, label_text, color in [
                (no_aero_data, no_aero_label, panel.get("no_aero_color", "#1f77b4")),
                (aero_data, aero_label, panel.get("aero_color", "#d62728")),
            ]:
                x = self.analyzer.channel_series(data, panel["x"])
                y = self.analyzer.channel_series(data, panel["y"])
                if x is None or y is None:
                    continue
                ax.scatter(x, y, alpha=0.4, s=25, label=label_text, color=color)
            ax.set_xlabel(panel.get("x_label", panel["x"]))
            ax.set_ylabel(panel.get("y_label", panel["y"]))
            ax.set_title(panel.get("title", self.graph["name"]), fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle(f"{self.graph['name']} - {plot_context}", fontsize=14, fontweight="bold")
        return self._save(fig, self._filename(f"{filename_suffix}_comparison"))

    def render_individual(self, label, data, config_label):
        data = self._filter_data(data)
        panels = self.graph.get("panels", [])
        fig, axes = plt.subplots(1, len(panels), figsize=tuple(self.graph.get("figsize", (14, 6))))
        axes = np.atleast_1d(axes)
        for ax, panel in zip(axes, panels):
            x = self.analyzer.channel_series(data, panel["x"])
            y = self.analyzer.channel_series(data, panel["y"])
            if x is None or y is None:
                ax.set_visible(False)
                continue
            ax.scatter(x, y, alpha=0.6, s=25, label=config_label)
            ax.set_xlabel(panel.get("x_label", panel["x"]))
            ax.set_ylabel(panel.get("y_label", panel["y"]))
            ax.set_title(panel.get("title", self.graph["name"]), fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle(f"{self.graph['name']} - {label}", fontsize=14, fontweight="bold")
        return self._save(fig, self._filename(label))


class SpeedBinnedScatterGraph(GraphRenderer):
    """Scatter plot split into panels by GPS speed ranges."""

    def render_comparison(self, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        no_aero_data = self._filter_data(self._combine_laps(no_aero_laps))
        aero_data = self._filter_data(self._combine_laps(aero_laps))
        speed_bins = self._speed_bins()
        fig, axes = self._create_axes(speed_bins)
        plotted = False
        point_sets = []

        for ax, speed_bin in zip(axes, speed_bins):
            for data, label_text, color in [
                (no_aero_data, no_aero_label, "#1f77b4"),
                (aero_data, aero_label, "#d62728"),
            ]:
                x, y = self._filtered_xy(data, speed_bin)
                if x is None or y is None or x.empty:
                    continue
                ax.scatter(x, y, alpha=0.4, s=self.graph.get("size", 18), label=label_text, color=color)
                point_sets.append((x, y))
                plotted = True
            self._finish_panel(ax, speed_bin)

        if not plotted:
            print(f"  Skipped {self.graph['name']}: missing x/y/speed channel or no points in speed ranges")
            plt.close(fig)
            return None

        self._apply_shared_axis_limits(axes, point_sets)
        fig.suptitle(f"{self.graph['name']} - {plot_context}", fontsize=14, fontweight="bold")
        return self._save(fig, self._filename(f"{filename_suffix}_comparison"))

    def render_individual(self, label, data, config_label):
        data = self._filter_data(data)
        speed_bins = self._speed_bins()
        fig, axes = self._create_axes(speed_bins)
        plotted = False
        point_sets = []

        for ax, speed_bin in zip(axes, speed_bins):
            x, y = self._filtered_xy(data, speed_bin)
            if x is None or y is None or x.empty:
                ax.set_visible(False)
                continue
            ax.scatter(x, y, alpha=0.55, s=self.graph.get("size", 18), label=config_label)
            point_sets.append((x, y))
            self._finish_panel(ax, speed_bin)
            plotted = True

        if not plotted:
            print(f"  Skipped {self.graph['name']} for {label}: missing x/y/speed channel or no points in speed ranges")
            plt.close(fig)
            return None

        self._apply_shared_axis_limits(axes, point_sets)
        fig.suptitle(f"{self.graph['name']} - {label}", fontsize=14, fontweight="bold")
        return self._save(fig, self._filename(label))

    def _create_axes(self, speed_bins):
        columns = min(2, len(speed_bins))
        rows = int(np.ceil(len(speed_bins) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=tuple(self.graph.get("figsize", (14, 10))))
        axes = np.atleast_1d(axes).flatten()
        for ax in axes[len(speed_bins):]:
            ax.set_visible(False)
        return fig, axes[:len(speed_bins)]

    def _speed_bins(self):
        return self.graph.get("speed_bins") or [{"min": 0, "max": 999, "label": "All speeds"}]

    def _filtered_xy(self, data, speed_bin):
        x = self.analyzer.channel_series(data, self.graph["x"])
        y = self.analyzer.channel_series(data, self.graph["y"])
        speed = self.analyzer.channel_series(data, self.graph.get("speed_channel", "gps_speed"))
        if x is None or y is None or speed is None:
            return None, None

        frame = pd.DataFrame({"x": x, "y": y, "speed": speed}).dropna()
        min_speed = speed_bin.get("min", -np.inf)
        max_speed = speed_bin.get("max", np.inf)
        mask = (frame["speed"] >= min_speed) & (frame["speed"] < max_speed)
        filtered = frame.loc[mask]
        return filtered["x"], filtered["y"]

    def _finish_panel(self, ax, speed_bin):
        ax.set_title(speed_bin.get("label", self._speed_bin_label(speed_bin)), fontsize=12, fontweight="bold")
        ax.set_xlabel(self.graph.get("x_label", self.graph["x"]))
        ax.set_ylabel(self.graph.get("y_label", self.graph["y"]))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    def _apply_shared_axis_limits(self, axes, point_sets):
        if self.graph.get("share_axis_limits", True) is False or not point_sets:
            return

        x_limits = self._padded_limits([points[0] for points in point_sets])
        y_limits = self._padded_limits([points[1] for points in point_sets])
        if x_limits is None or y_limits is None:
            return

        for ax in axes:
            if not ax.get_visible():
                continue
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)

    def _padded_limits(self, series_list):
        values = pd.concat(series_list, ignore_index=True)
        values = pd.to_numeric(values, errors="coerce").dropna()
        if values.empty:
            return None

        lower = values.min()
        upper = values.max()
        if lower == upper:
            padding = abs(lower) * 0.05 or 1.0
        else:
            padding = (upper - lower) * 0.05
        return lower - padding, upper + padding

    def _speed_bin_label(self, speed_bin):
        min_speed = speed_bin.get("min", "-inf")
        max_speed = speed_bin.get("max", "inf")
        return f"{min_speed}-{max_speed} km/h"


class HistogramPercentGraph(GraphRenderer):
    """Percent histogram for one or more channels."""

    def render_comparison(self, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        channels = self.graph.get("channels", [])
        fig, axes = plt.subplots(len(channels), 1, figsize=tuple(self.graph.get("figsize", (12, max(4, 3.5 * len(channels))))))
        axes = np.atleast_1d(axes)
        no_aero_data = self._filter_data(self._combine_laps(no_aero_laps))
        aero_data = self._filter_data(self._combine_laps(aero_laps))
        plotted = False
        for ax, channel in zip(axes, channels):
            no_aero_series = self.analyzer.channel_series(no_aero_data, channel["id"])
            aero_series = self.analyzer.channel_series(aero_data, channel["id"])
            bins = self.analyzer._combined_bins([no_aero_series, aero_series], bins=self.graph.get("bins", 12))
            if no_aero_series is not None:
                self.analyzer._draw_percent_bars(ax, no_aero_series, bins, "#1f77b4", no_aero_label, alpha=0.55)
                plotted = True
            if aero_series is not None:
                self.analyzer._draw_percent_bars(ax, aero_series, bins, "#d62728", aero_label, alpha=0.30, hatch="///", annotate=False)
                plotted = True
            ax.set_xlabel(channel.get("label", channel["id"]))
            ax.set_ylabel("Percent [%]")
            ax.grid(True, alpha=0.3)
            ax.legend()
        if not plotted:
            print(f"  Skipped {self.graph['name']}: no configured channels found")
            plt.close(fig)
            return None
        fig.suptitle(f"{self.graph['name']} - {plot_context}", fontsize=14, fontweight="bold")
        return self._save(fig, self._filename(f"{filename_suffix}_comparison"))

    def render_individual(self, label, data, config_label):
        data = self._filter_data(data)
        channels = self.graph.get("channels", [])
        fig, axes = plt.subplots(len(channels), 1, figsize=tuple(self.graph.get("figsize", (12, max(4, 3.5 * len(channels))))))
        axes = np.atleast_1d(axes)
        plotted = False
        for ax, channel in zip(axes, channels):
            series = self.analyzer.channel_series(data, channel["id"])
            if series is None:
                ax.set_visible(False)
                continue
            bins = self.analyzer._combined_bins([series], bins=self.graph.get("bins", 12))
            self.analyzer._draw_percent_bars(ax, series, bins, "#1f77b4", channel.get("label", channel["id"]))
            ax.set_xlabel(channel.get("label", channel["id"]))
            ax.set_ylabel("Percent [%]")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plotted = True
        if not plotted:
            print(f"  Skipped {self.graph['name']} for {label}: no configured channels found")
            plt.close(fig)
            return None
        fig.suptitle(f"{self.graph['name']} - {label}", fontsize=14, fontweight="bold")
        return self._save(fig, self._filename(label))


class ShockHistogramGraph(GraphRenderer):
    """Four-corner shock percent histogram for one configured shock metric."""

    shock_colors = {"FL": "#e41a1c", "FR": "#00c800", "RL": "#0000ff", "RR": "#ff7f0e"}

    def render_individual(self, label, data, config_label):
        data = self._filter_data(data)
        fig, axes = plt.subplots(2, 2, figsize=tuple(self.graph.get("figsize", (16, 12))))
        fig.subplots_adjust(top=0.86, hspace=0.55, wspace=0.20)
        for idx, shock in enumerate(self.graph["corners"]):
            ax = axes[idx // 2, idx % 2]
            self._draw_corner(ax, data, None, config_label, None, shock, comparison=False)
        fig.suptitle(f"{self.graph['name']} - {label}", fontsize=16, fontweight="bold")
        return self._save(fig, self._filename(label))

    def render_comparison(self, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        no_aero_data = self._filter_data(self._combine_laps(no_aero_laps))
        aero_data = self._filter_data(self._combine_laps(aero_laps))
        fig, axes = plt.subplots(2, 2, figsize=tuple(self.graph.get("figsize", (16, 12))))
        fig.subplots_adjust(top=0.86, hspace=0.55, wspace=0.20)
        for idx, shock in enumerate(self.graph["corners"]):
            ax = axes[idx // 2, idx % 2]
            self._draw_corner(ax, no_aero_data, aero_data, no_aero_label, aero_label, shock, comparison=True)
        fig.suptitle(f"{self.graph['name']} - {plot_context}", fontsize=16, fontweight="bold")
        return self._save(fig, self._filename(f"{filename_suffix}_comparison"))

    def _draw_corner(self, ax, primary_data, secondary_data, primary_label, secondary_label, shock, comparison):
        corner = shock["corner"]
        color = self.shock_colors.get(corner, "#1f77b4")
        channel_id = shock["channel"]
        primary_series = self.analyzer.channel_series(primary_data, channel_id)
        secondary_series = self.analyzer.channel_series(secondary_data, channel_id) if secondary_data is not None else None
        bins = self.analyzer._combined_bins([primary_series, secondary_series], bins=self.graph.get("bins", 10))
        if primary_series is not None:
            self.analyzer._draw_percent_bars(ax, primary_series, bins, color, primary_label, alpha=0.45 if comparison else 0.65)
        if secondary_series is not None:
            self.analyzer._draw_percent_bars(ax, secondary_series, bins, color, secondary_label, alpha=0.22, hatch="///", annotate=False)
        ax.set_title(f"{corner} {self.graph['name']}", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{corner} {self.graph.get('x_label', channel_id)}")
        ax.set_ylabel("Percent [%]")
        ax.grid(True, alpha=0.25)
        self._annotate_min_max(ax, [(primary_label, primary_series), (secondary_label, secondary_series)])
        ax.legend(fontsize=7, loc="upper left")

    def _annotate_min_max(self, ax, labelled_series):
        lines = []
        for label, series in labelled_series:
            if series is None:
                continue
            clean = series.dropna()
            if clean.empty:
                continue
            lines.append(f"{label}: min {clean.min():.2f}, max {clean.max():.2f}")
        if not lines:
            return
        ax.text(
            0.98,
            0.97,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85, "pad": 4},
        )


class TrackMapGraph(GraphRenderer):
    """GPS latitude/longitude track map."""

    def render_comparison(self, no_aero_laps, aero_laps, no_aero_label, aero_label, plot_context, filename_suffix):
        no_aero_laps = self._filtered_laps(no_aero_laps)
        aero_laps = self._filtered_laps(aero_laps)
        if not all(self.analyzer.has_channels(lap, ["gps_latitude", "gps_longitude"]) for lap in no_aero_laps + aero_laps):
            print("  Skipped track map: GPS latitude/longitude channels not found")
            return None
        no_aero_gps_laps = [self._gps_frame(lap) for lap in no_aero_laps]
        aero_gps_laps = [self._gps_frame(lap) for lap in aero_laps]
        no_aero_gps_laps = [lap for lap in no_aero_gps_laps if not lap.empty]
        aero_gps_laps = [lap for lap in aero_gps_laps if not lap.empty]
        if not no_aero_gps_laps or not aero_gps_laps:
            print("  Skipped track map: no valid GPS coordinates")
            return None
        origin_lat = pd.concat([self.analyzer.channel_series(lap, "gps_latitude") for lap in no_aero_gps_laps + aero_gps_laps], ignore_index=True).mean()
        origin_lon = pd.concat([self.analyzer.channel_series(lap, "gps_longitude") for lap in no_aero_gps_laps + aero_gps_laps], ignore_index=True).mean()
        fig, ax = plt.subplots(figsize=tuple(self.graph.get("figsize", (10, 10))))
        self._draw_laps(ax, no_aero_gps_laps, origin_lat, origin_lon, no_aero_label, "#1f77b4", "o")
        self._draw_laps(ax, aero_gps_laps, origin_lat, origin_lon, aero_label, "#d62728", "^")
        self._finish_axes(ax, f"{self.graph['name']} - {plot_context}")
        return self._save(fig, self._filename(f"{filename_suffix}_comparison"))

    def render_individual(self, label, data, config_label):
        data = self._filter_data(data)
        if not self.analyzer.has_channels(data, ["gps_latitude", "gps_longitude"]):
            print(f"  Skipped track map for {label}: GPS latitude/longitude channels not found")
            return None
        gps_data = self._gps_frame(data)
        if gps_data.empty:
            print(f"  Skipped track map for {label}: no valid GPS coordinates")
            return None
        origin_lat = self.analyzer.channel_series(gps_data, "gps_latitude").mean()
        origin_lon = self.analyzer.channel_series(gps_data, "gps_longitude").mean()
        fig, ax = plt.subplots(figsize=tuple(self.graph.get("figsize", (10, 10))))
        x, y = self.analyzer._latlon_to_local_xy(gps_data, origin_lat, origin_lon)
        ax.plot(x, y, label=config_label, color="#1f77b4", linewidth=2.0)
        ax.scatter(x.iloc[0], y.iloc[0], color="#1f77b4", marker="o", s=70, label="start")
        self._finish_axes(ax, f"{self.graph['name']} - {label}")
        return self._save(fig, self._filename(label))

    def _gps_frame(self, data):
        return data[[self.analyzer.resolve_channel(data, "gps_latitude"), self.analyzer.resolve_channel(data, "gps_longitude")]].dropna()

    def _draw_laps(self, ax, gps_laps, origin_lat, origin_lon, label, color, marker):
        first = True
        for gps_lap in gps_laps:
            x, y = self.analyzer._latlon_to_local_xy(gps_lap, origin_lat, origin_lon)
            ax.plot(x, y, label=label if first else None, color=color, linewidth=1.8, alpha=0.55)
            if first:
                ax.scatter(x.iloc[0], y.iloc[0], color=color, marker=marker, s=80, label=f"{label} start")
            first = False

    def _finish_axes(self, ax, title):
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("East/West Position (m)")
        ax.set_ylabel("North/South Position (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend()


class GraphRendererFactory:
    """Factory for graph renderer classes."""

    registry = {
        "timeseries": TimeSeriesGraph,
        "timeseries_grid": TimeSeriesGridGraph,
        "scatter": ScatterGraph,
        "scatter_grid": ScatterGridGraph,
        "speed_binned_scatter": SpeedBinnedScatterGraph,
        "histogram_percent": HistogramPercentGraph,
        "shock_histogram": ShockHistogramGraph,
        "track_map": TrackMapGraph,
    }

    @classmethod
    def create(cls, analyzer, graph):
        kind = graph.get("kind")
        renderer_cls = cls.registry.get(kind)
        if renderer_cls is None:
            raise ValueError(f"Unknown graph kind '{kind}' for graph '{graph.get('id')}'")
        return renderer_cls(analyzer, graph)
