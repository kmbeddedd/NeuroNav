from neuronav.visualization.forecast import plot_forecast_components, plot_3d_orbit_error
from neuronav.visualization.scientific import (
    plot_residual_qq,
    plot_training_history,
    plot_prediction_vs_actual,
    plot_multihorizon_heatmap,
    plot_residual_distributions,
    plot_per_satellite_mae,
    plot_probabilistic_uncertainty,
    plot_frequency_spectrum,
)

# Aliases
plot_predictions_vs_actual = plot_prediction_vs_actual
plot_multihorizon_mae_heatmap = plot_multihorizon_heatmap

__all__ = [
    'plot_forecast_components',
    'plot_3d_orbit_error',
    'plot_residual_qq',
    'plot_training_history',
    'plot_prediction_vs_actual',
    'plot_predictions_vs_actual',
    'plot_multihorizon_heatmap',
    'plot_multihorizon_mae_heatmap',
    'plot_residual_distributions',
    'plot_per_satellite_mae',
    'plot_probabilistic_uncertainty',
    'plot_frequency_spectrum',
]
