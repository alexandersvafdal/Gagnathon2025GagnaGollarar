import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# --- Matplotlib (SHAP) to HTML Image Converter ---
def convert_plt_to_base64(fig):
    buf = io.BytesIO()
    # --- MODIFIED: Save with a standard (white) background for the global plot ---
    # We will make the local plot transparent separately
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig) # Close the plot to save memory
    data = base64.b64encode(buf.getbuffer()).decode("utf8")
    return "data:image/png;base64," + data

# --- NEW: Converter for transparent (dark) plots ---
def convert_plt_to_base64_transparent(fig):
    buf = io.BytesIO()
    # Save with a transparent background
    fig.savefig(buf, format="png", bbox_inches='tight', transparent=True)
    plt.close(fig)
    data = base64.b64encode(buf.getbuffer()).decode("utf8")
    return "data:image/png;base64," + data

# --- (SIMULATION) Model & Data Generation (Identical) ---
@dash.callback(Output('models-store', 'data'), Input('models-store', 'id'))
def get_all_models_and_data(_):
    SECTORS = ["🐟 Cod Exports", "✈️ Tourism", "🏭 Aluminum"]
    models = {}
    explainers = {}
    historical_data = {}
    training_data = {}
    features_map = {
        "🐟 Cod Exports": ["Cod_Quota_kT", "Norwegian_Quota_kT", "ISK_EUR_Rate", "Marine_Fuel_Price"],
        "✈️ Tourism": ["Airline_Seat_Capacity", "ISK_EUR_Rate", "EU_Economic_Sentiment", "Jet_Fuel_Price"],
        "🏭 Aluminum": ["LME_Aluminum_Price", "Carbon_Credit_Price_EUR", "Global_Industrial_Production", "Electricity_Cost"]
    }
    
    for sector in SECTORS:
        features = features_map[sector]
        dates = pd.date_range(start="2018-01-01", end="2024-12-31", freq='MS')
        n = len(dates)
        data = pd.DataFrame(index=dates)
        X = pd.DataFrame(index=data.index[1:], columns=features)
        base_time = np.arange(n-1)
        seasonal = np.sin(base_time * 2 * np.pi / 12) * 50
        X[features[0]] = 100 + base_time * 0.1 + seasonal + np.random.randn(n-1)*5
        X[features[1]] = 120 - base_time * 0.05 + seasonal + np.random.randn(n-1)*5
        X[features[2]] = 140 + (np.random.randn(n-1)*1.5).cumsum()
        X[features[3]] = 80 + seasonal * 0.5 + np.random.randn(n-1)*2
        y = 50 + np.random.randn(n-1)*2
        y += (X[features[0]] - 100) * 2.5
        y -= (X[features[1]] - 120) * 2.0
        y -= (X[features[2]] - 140) * 1.5
        y -= (X[features[3]] - 80) * 0.5
        data['Target_Value'] = np.nan
        data.iloc[1:, data.columns.get_loc('Target_Value')] = y
        model = RandomForestRegressor(n_estimators=30, random_state=42)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        historical_data[sector] = data.to_json()
        training_data[sector] = X.to_json()

    return {
        "historical_data": historical_data,
        "training_data": training_data,
        "features_map": features_map,
        "sectors": SECTORS
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

sidebar = dbc.Col(
    [
        dbc.Card(
            dbc.CardBody([
                html.H4(html.I(className="fa-solid fa-industry me-2"), "1. Select Sector"),
                dcc.Dropdown(id='sector-dropdown', options=[], value=None, clearable=False, className='dark-dropdown'),
            ]),
            className="sidebar-card",
        ),
        dbc.Card(
            dbc.CardBody([
                html.H4(html.I(className="fa-solid fa-globe me-2"), "2. Macro-Economic Levers"),
                html.Label("Central Bank Rate (Rel. %)", className="form-label"),
                dcc.Slider(id='cbank-rate-slider', min=-2.0, max=2.0, step=0.25, value=0.0, marks={i: str(i) for i in [-2, 0, 2]}),
                
                html.Label("Brent Oil Price (% Change)", className="form-label"),
                dcc.Slider(id='oil-price-slider', min=-30.0, max=50.0, step=5.0, value=0.0, marks={i: f"{i}%" for i in [-30, 0, 50]}),

                html.Label("Wage Index (% Change)", className="form-label"),
                dcc.Slider(id='wage-index-slider', min=-5.0, max=10.0, step=1.0, value=0.0, marks={i: f"{i}%" for i in [-5, 0, 10]}),
            ]),
            className="sidebar-card",
        ),
        dbc.Card(
            dbc.CardBody([
                html.H4(html.I(className="fa-solid fa-sliders me-2"), "3. Sectoral Levers"),
                html.Div(id='sector-levers-container'),
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
            label="🔍 Explainability (SHAP)",
            tab_id="tab-explain",
            children=[
                html.H4("Global Feature Importance", className="mt-3"),
                html.P("Which features have the biggest impact on this model overall?"),
                # --- NEW: Added a wrapper with a light background ---
                html.Div(
                    html.Img(id='shap-global-plot', style={'width': '100%'}),
                    style={'backgroundColor': 'white', 'padding': '10px', 'borderRadius': '5px'}
                ),
                html.Hr(),
                html.H4("Local Forecast Explanation"),
                html.P("Why did the model predict a specific value for a single month?"),
                dcc.Dropdown(id='shap-month-dropdown', options=[], placeholder="Select a forecast month...", className='dark-dropdown'),
                html.Img(id='shap-local-plot', style={'width': '100%'})
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
    dcc.Store(id='models-store'),
    dcc.Store(id='simulation-data-store'),
])

# --- == Callbacks == ---

# Callback 1: Populate Sector Dropdown on Load
@app.callback(
    Output('sector-dropdown', 'options'),
    Output('sector-dropdown', 'value'),
    Input('models-store', 'data')
)
def populate_sector_dropdown(models_data):
    if models_data is None:
        return [], None
    sectors = models_data['sectors']
    options = [{'label': s, 'value': s} for s in sectors]
    return options, sectors[0]

# Callback 2: Update Sector-Specific Sliders
@app.callback(
    Output('sector-levers-container', 'children'),
    Input('sector-dropdown', 'value'),
    State('models-store', 'data')
)
def update_sectoral_levers(selected_sector, models_data):
    if not selected_sector or not models_data:
        return []
    features_map = models_data['features_map']
    sector_features = features_map[selected_sector]
    dynamic_sliders = []
    
    if "Cod_Quota_kT" in sector_features:
        dynamic_sliders.extend([
            html.Label("Icelandic Cod Quota (% Change)", className="form-label"),
            dcc.Slider(id={'type': 'sector-slider', 'feature': 'Cod_Quota_kT'}, min=-20, max=20, step=1, value=0, marks={-20: '-20%', 20: '+20%'})
        ])
    if "Norwegian_Quota_kT" in sector_features:
        dynamic_sliders.extend([
            html.Label("Norwegian/Russian Quota (% Change)", className="form-label"),
            dcc.Slider(id={'type': 'sector-slider', 'feature': 'Norwegian_Quota_kT'}, min=-20, max=20, step=1, value=0, marks={-20: '-20%', 20: '+20%'})
        ])
    if "Airline_Seat_Capacity" in sector_features:
        dynamic_sliders.extend([
            html.Label("Airline Seat Capacity (% Change)", className="form-label"),
            dcc.Slider(id={'type': 'sector-slider', 'feature': 'Airline_Seat_Capacity'}, min=-25, max=25, step=1, value=0, marks={-25: '-25%', 25: '+25%'})
        ])
    if "EU_Economic_Sentiment" in sector_features:
        dynamic_sliders.extend([
            html.Label("EU Economic Sentiment (% Change)", className="form-label"),
            dcc.Slider(id={'type': 'sector-slider', 'feature': 'EU_Economic_Sentiment'}, min=-10, max=10, step=1, value=0, marks={-10: '-10%', 10: '+10%'})
        ])
    if "LME_Aluminum_Price" in sector_features:
        dynamic_sliders.extend([
            html.Label("LME Aluminum Price (% Change)", className="form-label"),
            dcc.Slider(id={'type': 'sector-slider', 'feature': 'LME_Aluminum_Price'}, min=-30, max=30, step=1, value=0, marks={-30: '-30%', 30: '+30%'})
        ])
    if "Carbon_Credit_Price_EUR" in sector_features:
        dynamic_sliders.extend([
            html.Label("Carbon Credit Price (€/tonne)", className="form-label"),
            dcc.Slider(id={'type': 'sector-slider', 'feature': 'Carbon_Credit_Price_EUR'}, min=0, max=200, step=5, value=80, marks={0: '0€', 200: '200€'})
        ])
    
    dynamic_sliders.append(html.Br())
    return dynamic_sliders

# Callback 3: Run Simulation, Update Chart, and Update SHAP Plots
@app.callback(
    Output('main-forecast-chart', 'figure'),
    Output('simulation-data-store', 'data'),
    Output('shap-global-plot', 'src'),
    Input('sector-dropdown', 'value'),
    Input('cbank-rate-slider', 'value'),
    Input('oil-price-slider', 'value'),
    Input('wage-index-slider', 'value'),
    Input({'type': 'sector-slider', 'feature': ALL}, 'value'),
    State({'type': 'sector-slider', 'feature': ALL}, 'id'),
    State('models-store', 'data')
)
def run_simulation_and_update_plots(
    selected_sector, rate_mod, oil_mod, wage_mod,
    dynamic_slider_values, dynamic_slider_ids, models_data):

    if not selected_sector or not models_data:
        return go.Figure(layout={'template': 'plotly_dark'}), {}, ""

    sectoral_inputs = {}
    for i, slider_id_dict in enumerate(dynamic_slider_ids):
        feature_name = slider_id_dict['feature']
        feature_value = dynamic_slider_values[i]
        sectoral_inputs[feature_name] = feature_value

    features_map = models_data['features_map']
    sector_features = features_map[selected_sector]
    X_train_json = models_data['training_data'][selected_sector]
    hist_data_json = models_data['historical_data'][selected_sector]
    
    X_train = pd.read_json(X_train_json)
    hist_data = pd.read_json(hist_data_json)
    y_train = hist_data['Target_Value'].iloc[1:]
    
    model = RandomForestRegressor(n_estimators=30, random_state=42)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)

    future_dates = pd.date_range(start="2025-01-01", end="2026-12-31", freq='MS')
    n_future = len(future_dates)
    
    fx_rate_change = (wage_mod * 0.8) - (rate_mod * 2.0)
    isk_eur_rate_future = 145 * (1 + fx_rate_change / 100)
    marine_fuel_future = 100 * (1 + (oil_mod * 1.5) / 100)
    jet_fuel_future = 120 * (1 + (oil_mod * 1.2) / 100)
    eu_sentiment_future = 100 * (1 + (wage_mod / 100 * 0.5) - (oil_mod / 100 * 0.2))
    global_prod_future = 100 * (1 + (wage_mod / 100 * 0.3) - (oil_mod / 100 * 0.3))
    electricity_cost_future = 50 * (1 - rate_mod / 100)
    
    future_driver_values = {
        "ISK_EUR_Rate": np.full(n_future, isk_eur_rate_future),
        "Marine_Fuel_Price": np.full(n_future, marine_fuel_future),
        "Jet_Fuel_Price": np.full(n_future, jet_fuel_future),
        "EU_Economic_Sentiment": np.full(n_future, eu_sentiment_future),
        "Global_Industrial_Production": np.full(n_future, global_prod_future),
        "Electricity_Cost": np.full(n_future, electricity_cost_future),
        "seasonal_driver": np.sin(np.arange(n_future) * 2 * np.pi / 12) * 50
    }

    def build_x_future(inputs):
        X_future = pd.DataFrame(index=future_dates, columns=sector_features)
        for feature in sector_features:
            if feature in inputs:
                base_val = 110
                mod_val = inputs[feature]
                if feature == "Carbon_Credit_Price_EUR":
                    X_future[feature] = mod_val
                else:
                    X_future[feature] = base_val * (1 + mod_val / 100)
                if feature in ["Cod_Quota_kT", "Airline_Seat_Capacity", "LME_Aluminum_Price"]:
                    X_future[feature] += future_driver_values["seasonal_driver"]
            elif feature in future_driver_values:
                X_future[feature] = future_driver_values[feature]
            else:
                X_future[feature] = 100
        return X_future

    X_future_whatif = build_x_future(sectoral_inputs)
    predictions_whatif = model.predict(X_future_whatif)
    
    baseline_sector_inputs = {k: 0.0 for k in sectoral_inputs if k != "Carbon_Credit_Price_EUR"}
    if "Carbon_Credit_Price_EUR" in sectoral_inputs:
        baseline_sector_inputs["Carbon_Credit_Price_EUR"] = 80
    X_future_baseline = build_x_future(baseline_sector_inputs)
    predictions_baseline = model.predict(X_future_baseline)

    # --- 5. Generate Main Plotly Chart ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Target_Value'], mode='lines', name=f'Historical', line=dict(color='gray', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=future_dates, y=predictions_baseline, mode='lines', name='Baseline Forecast', line=dict(color='#007BFF', dash='dash', width=3)))
    fig.add_trace(go.Scatter(x=future_dates, y=predictions_whatif, mode='lines+markers', name='What-If Scenario', line=dict(color='#FF006E', width=4), marker=dict(size=5)))
    fig.update_layout(
        title=f"Forecast for {selected_sector.split(' ')[1]}",
        xaxis_title="Date",
        yaxis_title="Simulated Value",
        template="plotly_dark",
        legend_title="Series",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # --- 6. Generate Global SHAP Plot (REVERTED TO DEFAULT THEME) ---
    
    # --- NEW: Reset plt.rcParams to default for this plot ---
    plt.rcdefaults() 
    
    shap.summary_plot(
        explainer(X_train), X_train, 
        plot_type="bar", 
        show=False,
        # No color or theme arguments
    )
    
    fig_global = plt.gcf()
    # No custom title color
    fig_global.axes[0].set_title("Global Feature Importance")
    
    # --- Use the standard (white bg) converter ---
    global_plot_src = convert_plt_to_base64(fig_global)
    # --- END OF REVERT ---


    # --- 7. Prepare Data for Local SHAP Plot ---
    future_shap_values = explainer(X_future_whatif)
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]

    store_data = {
        "base_value": base_value,
        "shap_values": future_shap_values.values.tolist(),
        "x_future": X_future_whatif.to_json(),
        "features": sector_features,
        "predictions": predictions_whatif.tolist()
    }

    return fig, store_data, global_plot_src

# Callback 4: Update Local SHAP Dropdown
@app.callback(
    Output('shap-month-dropdown', 'options'),
    Output('shap-month-dropdown', 'value'),
    Input('simulation-data-store', 'data')
)
def update_month_dropdown(store_data):
    if not store_data:
        return [], None
    future_dates = pd.date_range(start="2025-01-01", end="2026-12-31", freq='MS')
    options = [{'label': d.strftime('%B %Y'), 'value': i} for i, d in enumerate(future_dates)]
    return options, 0

# Callback 5: Update Local SHAP Plot (Stays Dark)
@app.callback(
    Output('shap-local-plot', 'src'),
    Input('shap-month-dropdown', 'value'),
    State('simulation-data-store', 'data')
)
def update_local_shap_plot(month_idx, store_data):
    if month_idx is None or not store_data:
        return ""
        
    base_value = store_data['base_value']
    shap_values = np.array(store_data['shap_values'])
    X_future = pd.read_json(store_data['x_future'])
    features = store_data['features']
    
    shap_values_for_month = shap_values[month_idx]
    data_for_month = X_future.iloc[month_idx]

    # --- Set theme for *this plot only* ---
    theme_text_color = '#E9ECEF'
    plt.rcParams['text.color'] = theme_text_color
    plt.rcParams['axes.labelcolor'] = theme_text_color
    plt.rcParams['xtick.color'] = theme_text_color
    plt.rcParams['ytick.color'] = theme_text_color
    plt.rcParams['figure.facecolor'] = 'none'
    plt.rcParams['axes.facecolor'] = 'none'
    plt.rcParams['axes.edgecolor'] = theme_text_color
    plt.rcParams['savefig.transparent'] = True

    shap.waterfall_plot(
        shap.Explanation(
            base_values=base_value,
            values=shap_values_for_month,
            feature_names=features,
            data=data_for_month
        ),
        show=False
    )
    fig_local = plt.gcf()
    
    # --- Use the transparent converter ---
    local_plot_src = convert_plt_to_base64_transparent(fig_local)
    
    return local_plot_src

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=8050)