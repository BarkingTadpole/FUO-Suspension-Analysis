#!/usr/bin/env python3
"""Command-line app for aero vs no-aero FSAE telemetry comparison."""

import argparse
from pathlib import Path

from dashboard_config import DASHBOARD_GRAPHS
from telemetry_analysis import FSAETelemetryAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate aero vs no-aero telemetry overlay plots."
    )
    parser.add_argument(
        "--mode",
        choices=["comparison", "individual"],
        default="comparison",
        help="Dashboard mode. Default: comparison",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        default=None,
        help="Graph ids to generate. Use --list-graphs to see options. Default: all graphs",
    )
    parser.add_argument(
        "--list-graphs",
        action="store_true",
        help="Print available graph ids and exit.",
    )
    parser.add_argument(
        "--no-aero",
        nargs="+",
        default=["265.csv"],
        help="CSV file(s) for no-aero runs. Default: 265.csv",
    )
    parser.add_argument(
        "--aero",
        nargs="+",
        default=["266.csv", "273.csv"],
        help="CSV file(s) for aero runs. Default: 266.csv 273.csv",
    )
    parser.add_argument(
        "--data-dir",
        default=Path(__file__).parent,
        type=Path,
        help="Directory containing the CSV files. Default: this script's folder",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_graphs:
        print("Available graphs:")
        for graph in DASHBOARD_GRAPHS:
            print(f"  {graph['id']}: {graph['name']}")
        return

    no_aero_files = [args.data_dir / filename for filename in args.no_aero]
    aero_files = [args.data_dir / filename for filename in args.aero]
    if args.mode == "comparison":
        input_files = no_aero_files + aero_files
    else:
        input_files = no_aero_files + aero_files

    missing = [str(path) for path in input_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {', '.join(missing)}")

    analyzer = FSAETelemetryAnalyzer(args.data_dir, files=input_files, graph_ids=args.graphs)
    analyzer.analyze(mode=args.mode)


if __name__ == "__main__":
    main()
