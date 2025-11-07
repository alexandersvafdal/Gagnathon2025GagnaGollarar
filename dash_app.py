import dash
from dash import dcc, html, Input, Output, State, ALL
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

# --- 1. LOAD PRE-COMPUTED DATA ---
try:
    ALL_FORECASTS = pd.read_parquet("precomputed_forecasts.parquet")
    with open('baseline_shap_plots.json', 'r') as f:
        SHAP_PLOTS = json.load(f)
except FileNotFoundError:
    print("="*50)
    print("ERROR: 'precomputed_forecasts.parquet' or 'baseline_shap_plots.json' not found.")
    print("Please run 'python precompute.py' first!")
    print("="*50)
    exit()

# Extract scenario levels from the loaded data
SECTORS = ALL_FORECASTS[ALL_FORECASTS['sector'].notna()]['sector'].unique()
SCENARIO_LEVELS = {
    'rate_mod': sorted(ALL_FORECASTS['rate_mod'].dropna().unique()),
    'oil_mod': sorted(ALL_FORECASTS['oil_mod'].dropna().unique()),
    'wage_mod': sorted(ALL_FORECASTS['wage_mod'].dropna().unique()),
    'sector_mod_1': sorted(ALL_FORECASTS['sector_mod_1'].dropna().unique()),
    'sector_mod_2': sorted(ALL_FORECASTS['sector_mod_2'].dropna().unique()),
}

# --- Initialize the Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO, dbc.icons.FONT_AWESOME])
app.title = "Icelandic Economic Simulator"

# --- == Define the App Layout == ---
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="https.cdn-icons-png.flaticon.com/512/2170/2170316.png", height="30px")),
                        dbc.Col(dbc.NavbarBrand("Icelandic Economic Simulator")),
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

# --- Helper function to create RadioItems ---
def create_radio_options(values, unit='%'):
    """Helper to create labels for radio buttons."""
    # Use 'Baseline' for 0.0, and add units
    return [
        {"label": f"{v:+.1f}{unit}" if v != 0.0 else "Baseline (0.0%)", "value": v}
        for v in values
    ]

sidebar = dbc.Col(
    [
        dbc.Card(
            dbc.CardBody([
                html.H4(html.I(className="fa-solid fa-industry me-2"), "1. Select Sector"),
                dcc.Dropdown(id='sector-dropdown', options=SECTORS, value=SECTORS[0], clearable=False, className='dark-dropdown'),
            ]),
            className="sidebar-card",
        ),
        dbc.Card(
            dbc.CardBody([
                html.H4(html.I(className="fa-solid fa-globe me-2"), "2. Macro-Economic Scenarios"),
                
                html.Label("Central Bank Rate Scenario", className="form-label"),
                dbc.RadioItems(
                    id='rate-radio',
                    options=create_radio_options(SCENARIO_LEVELS['rate_mod']),
                    value=0.0,
                    inline=True, # <-- CHANGED
                    labelClassName="me-3", # margin
                ),
                
                html.Label("Brent Oil Price Scenario", className="form-label mt-3"),
                dbc.RadioItems(
                    id='oil-radio',
                    options=create_radio_options(SCENARIO_LEVELS['oil_mod']),
                    value=0.0,
                    inline=True, # <-- CHANGED
                    labelClassName="me-3",
                ),

                html.Label("Wage Index Scenario", className="form-label mt-3"),
                dbc.RadioItems(
                    id='wage-radio',
                    options=create_radio_options(SCENARIO_LEVELS['wage_mod']),
                    value=0.0,
                    inline=True, # <-- CHANGED
                    labelClassName="me-3",
                ),
            ]),
            className="sidebar-card",
        ),
        dbc.Card(
            dbc.CardBody([
                html.H4(html.I(className="fa-solid fa-sliders me-2"), "3. Sectoral Scenarios"),
                html.Div(id='sector-levers-container'), # Dynamic radio items will be injected here
            ]),
            className="sidebar-card",
        ),
    ],
    md=4,
)

main_content = dbc.Col(
    dbc.Tabs(id='main-tabs', children=[
        
        dbc.Tab(
            label="📈 Forecast & Scenario",
            tab_id="tab-forecast",
            children=[
                dcc.Graph(id='main-forecast-chart', figure={'layout': {'template': 'plotly_dark'}}, style={'height': '70vh'})
            ]
        ),
        
        dbc.Tab(
            label="🔍 Baseline Explainability",
            tab_id="tab-explain",
            children=[
                html.Div(className='mt-3', children=[
                    dbc.Alert("Explainability is shown for the 'Baseline' (all 0.0) scenario.", color="info"),
                    html.H4("Global Feature Importance (Baseline)"),
                    html.Div(
                        html.Img(id='shap-global-plot', style={'width': '100%'}),
                        style={'backgroundColor': 'white', 'padding': '10px', 'borderRadius': '5px'}
                    ),
                    html.Hr(),
                    html.H4("Local Forecast Explanation (Baseline, Jan 2025)"),
                    html.Img(id='shap-local-plot', style={'width': '100%'})
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

# Callback 1: Update Sector-Specific *RadioItems*
@app.callback(
    Output('sector-levers-container', 'children'),
    Input('sector-dropdown', 'value')
)
def update_sectoral_levers(selected_sector):
    if not selected_sector:
        return []

    # This maps the generic scenario name to a human-readable label
    sector_label_map = {
        "🐟 Cod Exports": {"mod1": "Icelandic Cod Quota", "mod2": "Norwegian/Russian Quota"},
        "✈️ Tourism": {"mod1": "Airline Seat Capacity", "mod2": "EU Economic Sentiment"},
        "🏭 Aluminum": {"mod1": "LME Aluminum Price", "mod2": "Carbon Credit Price"},
    }
    
    labels = sector_label_map[selected_sector]

    dynamic_components = [
        html.Label(f"{labels['mod1']} Scenario", className="form-label"),
        dbc.RadioItems(
            id='sector-mod-1-radio', 
            options=create_radio_options(SCENARIO_LEVELS['sector_mod_1']),
            value=0.0,
            inline=True, # <-- CHANGED
            labelClassName="me-3",
        ),
        
        html.Label(f"{labels['mod2']} Scenario", className="form-label mt-3"),
        dbc.RadioItems(
            id='sector-mod-2-radio', 
            options=create_radio_options(SCENARIO_LEVELS['sector_mod_2']),
            value=0.0,
            inline=True, # <-- CHANGED
            labelClassName="me-3",
        ),
        html.Br()
    ]
    
    return dynamic_components

# Callback 2: Update Main Chart based on RadioItem Filters
@app.callback(
    Output('main-forecast-chart', 'figure'),
    Input('sector-dropdown', 'value'),
    Input('rate-radio', 'value'),
    Input('oil-radio', 'value'),
    Input('wage-radio', 'value'),
    Input('sector-mod-1-radio', 'value'),
    Input('sector-mod-2-radio', 'value')
)
def update_chart_from_scenarios(sector, rate_mod, oil_mod, wage_mod, mod_1, mod_2):
    
    if mod_1 is None or mod_2 is None:
        return go.Figure(layout={'template': 'plotly_dark'})

    query_string = (
        f"sector == '{sector}' & "
        f"rate_mod == {rate_mod} & "
        f"oil_mod == {oil_mod} & "
        f"wage_mod == {wage_mod} & "
        f"sector_mod_1 == {mod_1} & "
        f"sector_mod_2 == {mod_2} & "
        f"forecast_value.notna()"
    )
    whatif_df = ALL_FORECASTS.query(query_string)

    baseline_query = (
        f"sector == '{sector}' & "
        f"rate_mod == 0.0 & oil_mod == 0.0 & wage_mod == 0.0 & "
        f"sector_mod_1 == 0.0 & sector_mod_2 == 0.0 & "
        f"forecast_value.notna()"
    )
    baseline_df = ALL_FORECASTS.query(baseline_query)
    
    hist_df = ALL_FORECASTS.query(f"sector == '{sector}' & historical_value.notna()")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['historical_value'], mode='lines', name=f'Historical', line=dict(color='gray', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=baseline_df['date'], y=baseline_df['forecast_value'], mode='lines', name='Baseline Forecast', line=dict(color='#007BFF', dash='dash', width=3)))
    fig.add_trace(go.Scatter(x=whatif_df['date'], y=whatif_df['forecast_value'], mode='lines+markers', name='Selected Scenario', line=dict(color='#FF006E', width=4), marker=dict(size=5)))
    
    fig.update_layout(
        title=f"Forecast for {sector.split(' ')[1]}",
        xaxis_title="Date",
        yaxis_title="Simulated Value",
        template="plotly_dark",
        legend_title="Series",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Callback 3: Update SHAP plots based on selected sector
@app.callback(
    Output('shap-global-plot', 'src'),
    Output('shap-local-plot', 'src'),
    Input('sector-dropdown', 'value')
)
def update_shap_plots(selected_sector):
    if not selected_sector or selected_sector not in SHAP_PLOTS:
        return "", ""
        
    global_plot_src = SHAP_PLOTS[selected_sector]['global_plot_src']
    local_plot_src = SHAP_PLOTS[selected_sector]['local_plot_src']
    
    return global_plot_src, local_plot_src

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=8050)