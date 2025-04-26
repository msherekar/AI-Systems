# Visualization utilities for model metrics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path

def plot_forecasts(actual, forecast, title='Forecast vs Actual', save_path=None):
    """
    Plot forecasted values against actual values.
    
    Args:
        actual (pd.Series): Actual values.
        forecast (pd.Series): Forecasted values.
        title (str): Plot title.
        save_path (str): Path to save the plot.
        
    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual.values, label='Actual')
    plt.plot(forecast.index, forecast.values, label='Forecast')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()

def plot_model_comparison(metrics, save_path=None):
    """
    Create a bar chart comparing model metrics.
    
    Args:
        metrics (dict): Dictionary of model metrics.
        save_path (str): Path to save the plot.
        
    Returns:
        None
    """
    models = list(metrics.keys())
    mse_values = [metrics[model]['mse'] for model in models]
    mae_values = [metrics[model]['mae'] for model in models]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('MSE Comparison', 'MAE Comparison'))
    
    fig.add_trace(
        go.Bar(x=models, y=mse_values, name='MSE'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=models, y=mae_values, name='MAE'),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Model Performance Comparison",
        height=500,
        width=1000,
        showlegend=False
    )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
    
    return fig

def plot_lstm_training_history(history, save_path=None):
    """
    Plot LSTM training history.
    
    Args:
        history (dict): Dictionary containing loss history.
        save_path (str): Path to save the plot.
        
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()

def create_interactive_forecast(dates, actual, forecast, title='Interactive Forecast', save_path=None):
    """
    Create an interactive Plotly chart for forecasts.
    
    Args:
        dates (list): List of dates.
        actual (list): Actual values.
        forecast (list): Forecasted values.
        title (str): Plot title.
        save_path (str): Path to save the plot.
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines',
        name='Actual'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=forecast,
        mode='lines',
        name='Forecast'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        legend_title='Data Type',
        height=600,
        width=1000
    )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
    
    return fig 