import dash
from dash import Dash,dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
# Note: We no longer need shap or sklearn in the dashboard itself
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os
debug = True


# --- LOAD HISTORICAL DATA AND PREDICTIONS ---
try:
    HISTORICAL_DATA = pd.read_parquet('../historical_data.parquet')
    ALL_PREDICTIONS = pd.read_parquet('../all_predictions.parquet')
    
    # Combine historical and predictions for easy filtering
    COMBINED_DATA = pd.concat([HISTORICAL_DATA, ALL_PREDICTIONS], ignore_index=True)
    
    # Get unique countries and branches
    ALL_COUNTRIES = sorted(COMBINED_DATA['Country'].dropna().unique().tolist())
    ALL_BRANCHES = sorted(COMBINED_DATA['Branches'].dropna().unique().tolist())
    
    print(f"Loaded {len(HISTORICAL_DATA)} historical records and {len(ALL_PREDICTIONS)} predictions")
    print(f"Available countries: {len(ALL_COUNTRIES)}, Available branches: {len(ALL_BRANCHES)}")
except FileNotFoundError as e:
    print("="*50)
    print(f"ERROR: Required data files not found: {e}")
    print("Please ensure 'historical_data.parquet' and 'all_predictions.parquet' exist.")
    print("="*50)
    exit()

# --- Initialize the Dash App ---
# REMOVED the first, incorrect app = Dash(__name__)
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.SUPERHERO, dbc.icons.FONT_AWESOME]
)

# Ensure the `server` attribute is exposed for Gunicorn
server = app.server

# --- == Define the App Layout == ---
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="https.cdn-icons-png.flaticon.com/512/2170/2170316.png", height="30px")),
                        dbc.Col(dbc.NavbarBrand("Icelandic Export Data & Predictions")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="#",
                style={"textDecoration": "none"},
            ),
        ]
    ),
    color="dark",
    dark=True,
    className="mb-4",
)

sidebar = dbc.Col(
    [
        dbc.Card(
            dbc.CardBody([
                html.H4(html.I(className="fa-solid fa-flag me-2"), "Select Export Data"),
                
                html.Label("Country", className="form-label"),
                dcc.Dropdown(
                    id='country-dropdown',
                    options=[{'label': country, 'value': country} for country in ALL_COUNTRIES],
                    value=ALL_COUNTRIES[0] if ALL_COUNTRIES else None,
                    clearable=False,
                    className='dark-dropdown'
                ),
                
                html.Label("Branch / Product Category", className="form-label mt-3"),
                dcc.Dropdown(
                    id='branch-dropdown',
                    options=[],  # Will be populated dynamically
                    value=None,
                    clearable=False,
                    className='dark-dropdown'
                ),
            ]),
            className="sidebar-card",
        ),
    ],
    md=4,
)

main_content = dbc.Col(
    dbc.Tabs(id='main-tabs', children=[
        
        dbc.Tab(
            label="📈 Time Series",
            tab_id="tab-forecast",
            children=[
                dcc.Graph(id='main-forecast-chart', figure={'layout': {'template': 'plotly_dark'}}, style={'height': '70vh'})
            ]
        ),
        
        dbc.Tab(
            label="📊 Prediction Breakdown",
            tab_id="tab-export",
            children=[
                html.Div(className='mt-3', children=[
                    dcc.Graph(id='export-chart', figure={'layout': {'template': 'plotly_dark'}}, style={'height': '70vh'})
                ])
            ]
        ),
        
        dbc.Tab(
            label="🎯 Model Accuracy",
            tab_id="tab-accuracy",
            children=[
                html.Div(className='mt-3', children=[
                    dcc.Graph(id='accuracy-chart', figure={'layout': {'template': 'plotly_dark'}}, style={'height': '70vh'})
                ])
            ]
        ),
        
        dbc.Tab(
            label="📉 Residuals Analysis",
            tab_id="tab-residuals",
            children=[
                html.Div(className='mt-3', children=[
                    dcc.Graph(id='residuals-chart', figure={'layout': {'template': 'plotly_dark'}}, style={'height': '70vh'})
                ])
            ]
        ),
        
        dbc.Tab(
            label="🌍 Country Comparison",
            tab_id="tab-comparison",
            children=[
                html.Div(className='mt-3', children=[
                    html.Label("Select up to 5 countries to compare:", className="form-label"),
                    dcc.Dropdown(
                        id='comparison-countries',
                        options=[{'label': country, 'value': country} for country in ALL_COUNTRIES],
                        value=[ALL_COUNTRIES[0]] if ALL_COUNTRIES else [],
                        multi=True,
                        className='dark-dropdown mb-3'
                    ),
                    dcc.Graph(id='comparison-chart', figure={'layout': {'template': 'plotly_dark'}}, style={'height': '60vh'})
                ])
            ]
        ),
    ]),
    md=8,
)

app.layout = html.Div([
    navbar,
    dbc.Container(
        dbc.Row([sidebar, main_content]),
        fluid=True,
        className="dbc"
    ),
])

# --- == Callbacks == ---

# Callback 0: Update Branch dropdown based on selected Country
@app.callback(
    Output('branch-dropdown', 'options'),
    Output('branch-dropdown', 'value'),
    Input('country-dropdown', 'value')
)
def update_branch_dropdown(selected_country):
    if not selected_country or COMBINED_DATA is None:
        return [], None
    
    # Filter branches that exist for the selected country
    filtered_branches = COMBINED_DATA[COMBINED_DATA['Country'] == selected_country]['Branches'].dropna().unique()
    branch_options = [{'label': branch, 'value': branch} for branch in sorted(filtered_branches)]
    
    # Set the first branch as default value
    default_value = sorted(filtered_branches)[0] if len(filtered_branches) > 0 else None
    
    return branch_options, default_value

# Callback 1: Update Main Chart based on Country/Branch selection
@app.callback(
    Output('main-forecast-chart', 'figure'),
    Input('country-dropdown', 'value'),
    Input('branch-dropdown', 'value')
)
def update_chart_from_scenarios(selected_country, selected_branch):
    
    fig = go.Figure()
    
    if not selected_country or not selected_branch:
        fig.update_layout(
            title="Please select a Country and Branch",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    # Add historical data
    historical_filtered = HISTORICAL_DATA[
        (HISTORICAL_DATA['Country'] == selected_country) & 
        (HISTORICAL_DATA['Branches'] == selected_branch)
    ].copy()
    
    # Add predictions
    predictions_filtered = ALL_PREDICTIONS[
        (ALL_PREDICTIONS['Country'] == selected_country) & 
        (ALL_PREDICTIONS['Branches'] == selected_branch)
    ].copy()
    
    if not predictions_filtered.empty:
        predictions_filtered['timestamp'] = pd.to_datetime(predictions_filtered['timestamp'])
        predictions_filtered = predictions_filtered.sort_values('timestamp')
        
        # Separate by prediction type
        validation = predictions_filtered[predictions_filtered['prediction_type'] == 'Validation']
        future = predictions_filtered[predictions_filtered['prediction_type'] == 'Future']
        
        # Combine historical data with validation actual values for continuous line
        actual_data = []
        
        if not historical_filtered.empty:
            historical_filtered['timestamp'] = pd.to_datetime(historical_filtered['timestamp'])
            actual_data.append(historical_filtered[['timestamp', 'y_true']])
        
        if not validation.empty:
            actual_data.append(validation[['timestamp', 'y_true']])
        
        if actual_data:
            combined_actual = pd.concat(actual_data, ignore_index=True)
            combined_actual = combined_actual.sort_values('timestamp')
            
            fig.add_trace(go.Scatter(
                x=combined_actual['timestamp'], 
                y=combined_actual['y_true'], 
                mode='lines', 
                name='Historical Data', 
                line=dict(color='#808080', width=2, dash='dot'),
                opacity=0.7
            ))
        
        # Add validation predictions
        if not validation.empty:
            fig.add_trace(go.Scatter(
                x=validation['timestamp'],
                y=validation['y_pred'],
                mode='lines+markers',
                name='Validation Prediction',
                line=dict(color='#FFD700', width=2),
                marker=dict(size=4)
            ))
        
        if not future.empty:
            fig.add_trace(go.Scatter(
                x=future['timestamp'],
                y=future['y_pred'],
                mode='lines+markers',
                name='Future Forecast',
                line=dict(color='#00FF00', width=3),
                marker=dict(size=6)
            ))
    
    title_text = f"{selected_country}: {selected_branch}"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_dark",
        legend_title="Series",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Callback 2: Update Export Chart based on Country and Branch
@app.callback(
    Output('export-chart', 'figure'),
    Input('country-dropdown', 'value'),
    Input('branch-dropdown', 'value')
)
def update_export_chart(selected_country, selected_branch):
    if not selected_country or not selected_branch:
        return go.Figure(layout={'template': 'plotly_dark', 'title': 'No data available'})
    
    # Get historical data
    historical_filtered = HISTORICAL_DATA[
        (HISTORICAL_DATA['Country'] == selected_country) & 
        (HISTORICAL_DATA['Branches'] == selected_branch)
    ].copy()
    
    # Get predictions
    predictions_filtered = ALL_PREDICTIONS[
        (ALL_PREDICTIONS['Country'] == selected_country) & 
        (ALL_PREDICTIONS['Branches'] == selected_branch)
    ].copy()
    
    if historical_filtered.empty and predictions_filtered.empty:
        return go.Figure(layout={'template': 'plotly_dark', 'title': 'No data for selected filters'})
    
    # Create figure
    fig = go.Figure()
    
    # Add predictions
    if not predictions_filtered.empty:
        predictions_filtered['timestamp'] = pd.to_datetime(predictions_filtered['timestamp'])
        predictions_filtered = predictions_filtered.sort_values('timestamp')
        
        # Separate by prediction type
        validation = predictions_filtered[predictions_filtered['prediction_type'] == 'Validation']
        future = predictions_filtered[predictions_filtered['prediction_type'] == 'Future']
        
        # Combine historical data with validation actual values for continuous line
        actual_data = []
        
        if not historical_filtered.empty:
            historical_filtered['timestamp'] = pd.to_datetime(historical_filtered['timestamp'])
            actual_data.append(historical_filtered[['timestamp', 'y_true']])
        
        if not validation.empty:
            actual_data.append(validation[['timestamp', 'y_true']])
        
        if actual_data:
            combined_actual = pd.concat(actual_data, ignore_index=True)
            combined_actual = combined_actual.sort_values('timestamp')
            
            fig.add_trace(go.Scatter(
                x=combined_actual['timestamp'],
                y=combined_actual['y_true'],
                mode='lines',
                name='Actual Data',
                line=dict(color='#00D9FF', width=3),
                opacity=0.8
            ))
        
        # Add validation predictions
        if not validation.empty:
            fig.add_trace(go.Scatter(
                x=validation['timestamp'],
                y=validation['y_pred'],
                mode='lines+markers',
                name='Validation Prediction',
                line=dict(color='#FFD700', width=2, dash='dash'),
                marker=dict(size=4)
            ))
        
        if not future.empty:
            fig.add_trace(go.Scatter(
                x=future['timestamp'],
                y=future['y_pred'],
                mode='lines+markers',
                name='Future Forecast',
                line=dict(color='#00FF00', width=3),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title=f"Export Data & Predictions: {selected_country} - {selected_branch}",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Value"),
        template="plotly_dark",
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Callback 3: Model Accuracy Chart
@app.callback(
    Output('accuracy-chart', 'figure'),
    Input('country-dropdown', 'value'),
    Input('branch-dropdown', 'value')
)
def update_accuracy_chart(selected_country, selected_branch):
    if not selected_country or not selected_branch:
        return go.Figure(layout={'template': 'plotly_dark', 'title': 'Please select a Country and Branch'})
    
    # Get predictions with actual values for comparison
    predictions_filtered = ALL_PREDICTIONS[
        (ALL_PREDICTIONS['Country'] == selected_country) & 
        (ALL_PREDICTIONS['Branches'] == selected_branch) &
        (ALL_PREDICTIONS['y_true'].notna())
    ].copy()
    
    if predictions_filtered.empty:
        return go.Figure(layout={'template': 'plotly_dark', 'title': 'No validation/test data available'})
    
    predictions_filtered['timestamp'] = pd.to_datetime(predictions_filtered['timestamp'])
    predictions_filtered = predictions_filtered.sort_values('timestamp')
    
    # Create scatter plot: actual vs predicted
    fig = go.Figure()
    
    # Add perfect prediction line
    min_val = min(predictions_filtered['y_true'].min(), predictions_filtered['y_pred'].min())
    max_val = max(predictions_filtered['y_true'].max(), predictions_filtered['y_pred'].max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=True
    ))
    
    # Separate by prediction type
    for pred_type, color in [('Validation', '#FFD700')]:
        subset = predictions_filtered[predictions_filtered['prediction_type'] == pred_type]
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset['y_true'],
                y=subset['y_pred'],
                mode='markers',
                name=pred_type,
                marker=dict(size=8, color=color, opacity=0.6),
                text=subset['timestamp'].dt.strftime('%Y-%m'),
                hovertemplate='<b>%{text}</b><br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
            ))
    
    fig.update_layout(
        title=f"Model Accuracy: {selected_country} - {selected_branch}",
        xaxis=dict(title="Actual Value"),
        yaxis=dict(title="Predicted Value"),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='closest'
    )
    
    return fig

# Callback 4: Residuals Analysis Chart
@app.callback(
    Output('residuals-chart', 'figure'),
    Input('country-dropdown', 'value'),
    Input('branch-dropdown', 'value')
)
def update_residuals_chart(selected_country, selected_branch):
    if not selected_country or not selected_branch:
        return go.Figure(layout={'template': 'plotly_dark', 'title': 'Please select a Country and Branch'})
    
    # Get predictions with actual values
    predictions_filtered = ALL_PREDICTIONS[
        (ALL_PREDICTIONS['Country'] == selected_country) & 
        (ALL_PREDICTIONS['Branches'] == selected_branch) &
        (ALL_PREDICTIONS['y_true'].notna())
    ].copy()
    
    if predictions_filtered.empty:
        return go.Figure(layout={'template': 'plotly_dark', 'title': 'No validation/test data available'})
    
    predictions_filtered['timestamp'] = pd.to_datetime(predictions_filtered['timestamp'])
    predictions_filtered = predictions_filtered.sort_values('timestamp')
    predictions_filtered['residual'] = predictions_filtered['y_true'] - predictions_filtered['y_pred']
    
    fig = go.Figure()
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Error")
    
    # Plot residuals by prediction type
    for pred_type, color in [('Validation', '#FFD700')]:
        subset = predictions_filtered[predictions_filtered['prediction_type'] == pred_type]
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset['timestamp'],
                y=subset['residual'],
                mode='markers+lines',
                name=pred_type,
                marker=dict(size=6, color=color),
                line=dict(color=color, width=1)
            ))
    
    fig.update_layout(
        title=f"Prediction Residuals: {selected_country} - {selected_branch}",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Residual (Actual - Predicted)"),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    
    return fig

# Callback 5: Country Comparison Chart
@app.callback(
    Output('comparison-chart', 'figure'),
    Input('comparison-countries', 'value'),
    Input('branch-dropdown', 'value')
)
def update_comparison_chart(selected_countries, selected_branch):
    if not selected_countries or not selected_branch:
        return go.Figure(layout={'template': 'plotly_dark', 'title': 'Please select countries and a branch'})
    
    # Limit to 5 countries
    if len(selected_countries) > 5:
        selected_countries = selected_countries[:5]
    
    fig = go.Figure()
    
    colors = ['#00D9FF', '#FF006E', '#00FF00', '#FFD700', '#FF8C00']
    
    for idx, country in enumerate(selected_countries):
        # Get historical data
        hist_data = HISTORICAL_DATA[
            (HISTORICAL_DATA['Country'] == country) & 
            (HISTORICAL_DATA['Branches'] == selected_branch)
        ].copy()
        
        if not hist_data.empty:
            hist_data['timestamp'] = pd.to_datetime(hist_data['timestamp'])
            hist_data = hist_data.sort_values('timestamp')
            
            fig.add_trace(go.Scatter(
                x=hist_data['timestamp'],
                y=hist_data['y_true'],
                mode='lines',
                name=country,
                line=dict(color=colors[idx % len(colors)], width=2),
                opacity=0.8
            ))
        
        # Get forecast data
        forecast_data = ALL_PREDICTIONS[
            (ALL_PREDICTIONS['Country'] == country) & 
            (ALL_PREDICTIONS['Branches'] == selected_branch) &
            (ALL_PREDICTIONS['prediction_type'] == 'Future')
        ].copy()
        
        if not forecast_data.empty:
            forecast_data['timestamp'] = pd.to_datetime(forecast_data['timestamp'])
            forecast_data = forecast_data.sort_values('timestamp')
            
            fig.add_trace(go.Scatter(
                x=forecast_data['timestamp'],
                y=forecast_data['y_pred'],
                mode='lines',
                name=f'{country} (Forecast)',
                line=dict(color=colors[idx % len(colors)], width=2, dash='dash'),
                opacity=0.6,
                showlegend=True
            ))
    
    fig.update_layout(
        title=f"Country Comparison: {selected_branch}",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Value"),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig



if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8050", debug=debug)