#!/usr/bin/env python3
"""Test script for telemetry analysis - runs in comparison mode"""

from pathlib import Path
from telemetry_analysis import FSAETelemetryAnalyzer

data_dir = Path(__file__).parent
analyzer = FSAETelemetryAnalyzer(data_dir)
analyzer.analyze(mode='comparison')
