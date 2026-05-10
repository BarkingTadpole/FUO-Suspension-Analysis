"""Dashboard graph registry.

To rename or reorder graphs, edit DASHBOARD_GRAPHS.
To add a new graph, add a new entry here and implement the matching renderer
method in telemetry_analysis.py.
"""

DASHBOARD_TITLE = "FSAE Telemetry Dashboard"

DASHBOARD_GRAPHS = [
    {
        "id": "shock",
        "name": "Shock Travel & Velocity",
        "comparison_method": "_plot_shock_histograms_comparison",
        "individual_method": "_plot_shock_histograms",
    },
    {
        "id": "speed_yaw_rpm",
        "name": "GPS Speed, Yaw Rate & ECU RPM",
        "comparison_method": "_plot_speed_yaw_rpm_comparison",
        "individual_method": "_plot_speed_yaw_rpm",
    },
    {
        "id": "speed_yaw_rpm_histograms",
        "name": "GPS Speed, Yaw Rate & ECU RPM Percent Histograms",
        "comparison_method": "_plot_speed_yaw_rpm_histograms_comparison",
        "individual_method": None,
    },
    {
        "id": "gg",
        "name": "GG Diagram",
        "comparison_method": "_plot_gg_diagram_comparison",
        "individual_method": "_plot_gg_diagram",
    },
    {
        "id": "speed_yaw",
        "name": "GPS Speed vs Yaw Rate",
        "comparison_method": "_plot_speed_vs_yaw_comparison",
        "individual_method": "_plot_speed_vs_yaw",
    },
    {
        "id": "latacc_yaw",
        "name": "GPS LatAcc vs Yaw Rate",
        "comparison_method": "_plot_latacc_vs_yaw_comparison",
        "individual_method": "_plot_latacc_vs_yaw",
    },
    {
        "id": "speed_wheelspeed",
        "name": "GPS Speed vs Wheel Speed",
        "comparison_method": "_plot_speed_vs_wheelspeed_comparison",
        "individual_method": "_plot_speed_vs_wheelspeed",
    },
    {
        "id": "track_map",
        "name": "GPS Track Map",
        "comparison_method": "_plot_track_map_comparison",
        "individual_method": "_plot_track_map_individual",
    },
]
