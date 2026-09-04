"""Realtime and GUI forecast visualization routines for NeuroNav."""
from __future__ import annotations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

def plot_forecast_components(
    df: pd.DataFrame,
    satellite_id: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    show_uncertainty: bool = True
) -> plt.Figure:
    """Plot predicted Error_X, Error_Y, Error_Z, Error_Clock, and 3D Orbit Error."""
    plot_df = df.copy()
    if satellite_id is not None and 'Satellite_ID' in plot_df.columns:
        plot_df = plot_df[plot_df['Satellite_ID'] == satellite_id]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    time_col = 'utc_time' if 'utc_time' in plot_df.columns else plot_df.index
    times = pd.to_datetime(time_col)

    targets = [
        ('Error_X', 'pred_Error_X', 'X Error (m)'),
        ('Error_Y', 'pred_Error_Y', 'Y Error (m)'),
        ('Error_Z', 'pred_Error_Z', 'Z Error (m)'),
        ('Error_Clock', 'pred_Error_Clock', 'Clock Error (m)'),
    ]

    for idx, (actual_col, pred_col, label) in enumerate(targets):
        ax = axes[idx]
        if actual_col in plot_df.columns:
            ax.plot(times, plot_df[actual_col], 'k-', label='Actual', alpha=0.7, lw=1.5)
        if pred_col in plot_df.columns:
            ax.plot(times, plot_df[pred_col], 'r--', label='Predicted', lw=1.8)
            low_col = f'{pred_col}_low'
            high_col = f'{pred_col}_high'
            if show_uncertainty and low_col in plot_df.columns and high_col in plot_df.columns:
                ax.fill_between(times, plot_df[low_col], plot_df[high_col], color='red', alpha=0.2, label='90% Uncertainty')

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            sat_str = f' - Satellite {satellite_id}' if satellite_id else ''
            ax.set_title(f'GNSS Orbit and Clock Error Forecast{sat_str}', fontweight='bold')
            ax.legend(loc='upper right')

    axes[-1].set_xlabel('UTC Time')
    fig.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches='tight')
    return fig

def plot_3d_orbit_error(
    df: pd.DataFrame,
    satellite_id: Optional[str] = None,
    output_path: Optional[str | Path] = None
) -> plt.Figure:
    """Plot 3D Orbit Error magnitude over forecast horizon."""
    plot_df = df.copy()
    if satellite_id is not None and 'Satellite_ID' in plot_df.columns:
        plot_df = plot_df[plot_df['Satellite_ID'] == satellite_id]

    fig, ax = plt.subplots(figsize=(10, 4))
    time_col = 'utc_time' if 'utc_time' in plot_df.columns else plot_df.index
    times = pd.to_datetime(time_col)

    if 'pred_3D_Orbit_Error' in plot_df.columns:
        ax.plot(times, plot_df['pred_3D_Orbit_Error'], 'b-', label='Predicted 3D Error', lw=2.0)
    elif 'pred_orbit_3d_error' in plot_df.columns:
        ax.plot(times, plot_df['pred_orbit_3d_error'], 'b-', label='Predicted 3D Error', lw=2.0)

    if '3D_Orbit_Error' in plot_df.columns:
        ax.plot(times, plot_df['3D_Orbit_Error'], 'k--', label='Actual 3D Error', alpha=0.6, lw=1.2)

    ax.set_ylabel('3D Orbit Error (m)')
    ax.set_xlabel('UTC Time')
    sat_str = f' ({satellite_id})' if satellite_id else ''
    ax.set_title(f'3D Orbit Error Norm over Time{sat_str}', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    fig.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches='tight')
    return fig
