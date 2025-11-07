# --- START OF FIX ---
# Force matplotlib to use a non-GUI backend (Agg)
# This MUST be done before importing pyplot or shap
import matplotlib
matplotlib.use('Agg')
# --- END OF FIX ---

import dash
from dash import dcc, html, Input, Output, State, ALL
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt # This import is now safe
import io
import base64

# --- Matplotlib (SHAP) to HTML Image Converter ---
def convert_plt_to_base64(fig):
    """Converts a matplotlib figure to a base64 string for html.Img."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig) # Close the plot to save memory
    data = base64.b64encode(buf.getbuffer()).decode("utf8")
    return "data:image/png;base64," + data

# --- (SIMULATION) Model & Data Generation ---
# This is the same "backend" from our Streamlit app
@dash.callback(Output('models-store', 'data'), Input('models-store', 'id')) # This is a trick to run on load
def get_all_models_and_data(_):
    """
    Simulates a multi-model ecosystem for different sectors.
    This runs ONCE on startup and stores models in dcc.Store.
    NOTE: Storing trained models like this is simple for a demo,
    but for true production, you'd load them from disk.
    """
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
    
    # --- Generate a model for each sector ---
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
        
        # We can't store the actual model objects in dcc.Store (not JSON serializable)
        # For this demo, we'll just re-train the simple model inside the callback.
        # In a real app, you'd load a saved model from disk.
        
        # Store data needed to re-train
        historical_data[sector] = data.to_json()
        training_data[sector] = X.to_json()

    # We will return the *data* and *features*, and re-train the
    # simple model on-the-fly inside the callback.
    return {
        "historical_data": historical_data,
        "training_data": training_data,
        "features_map": features_map,
        "sectors": SECTORS
    }

# --- Initialize the Dash App ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = "Icelandic Economic Simulator"

# --- Define the App Layout ---
app.layout = html.Div([
    
    # --- Invisible Data Stores ---
    # Stores the "global" model data (runs once on load)
    dcc.Store(id='models-store'),
    # Stores the *results* of the latest simulation
    dcc.Store(id='simulation-data-store'),

    # --- Header ---
    html.H1("🌍 Icelandic Macro-Economic Simulator (Dash Version)", style={'textAlign': 'center'}),
    html.Hr(),

    # --- Main Container (Sidebar + Content) ---
    html.Div(className='row', children=[
        
        # --- Sidebar ---
        html.Div(className='three columns', style={'padding': '10px'}, children=[
            html.H3("1. Select Sector"),
            dcc.Dropdown(
                id='sector-dropdown',
                options=[], # Will be populated by callback
                value=None,
                clearable=False
            ),
            
            html.H3("2. 🌍 Macro-Economic Levers"),
            html.Label("Central Bank Rate (Rel. %)"),
            dcc.Slider(id='cbank-rate-slider', min=-2.0, max=2.0, step=0.25, value=0.0, marks={i: str(i) for i in [-2, 0, 2]}),
            
            html.Label("Brent Oil Price (% Change)"),
            dcc.Slider(id='oil-price-slider', min=-30.0, max=50.0, step=5.0, value=0.0, marks={i: f"{i}%" for i in [-30, 0, 50]}),
            
            html.Label("Wage Index (% Change)"),
            dcc.Slider(id='wage-index-slider', min=-5.0, max=10.0, step=1.0, value=0.0, marks={i: f"{i}%" for i in [-5, 0, 10]}),
            
            html.Hr(),
            
            html.H3("3. Sectoral Levers"),
            # This Div is the *target* for our dynamic sliders
            html.Div(id='sector-levers-container'),
        ]),
        
        # --- Main Content Area (Tabs) ---
        html.Div(className='nine columns', children=[
            dcc.Tabs(id='main-tabs', children=[
                
                # --- Tab 1: Forecast Chart ---
                dcc.Tab(label='📈 Forecast & Scenario', children=[
                    dcc.Graph(id='main-forecast-chart')
                ]),
                
                # --- Tab 2: Explainability (SHAP) ---
                dcc.Tab(label='🔍 Explainability (SHAP)', children=[
                    html.Div(style={'padding': '10px'}, children=[
                        html.H4("Global Feature Importance"),
                        html.P("Which features have the biggest impact on this model overall?"),
                        # SHAP plots will be rendered as images
                        html.Img(id='shap-global-plot', style={'width': '100%'}),
                        
                        html.Hr(),
                        
                        html.H4("Local Forecast Explanation"),
                        html.P("Why did the model predict a specific value for a single month?"),
                        dcc.Dropdown(id='shap-month-dropdown', options=[], placeholder="Select a forecast month..."),
                        html.Img(id='shap-local-plot', style={'width': '100%'})
                    ])
                ]),
            ])
        ]),
    ]),
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
    return options, sectors[0] # Return options and default value

# Callback 2: Update Sector-Specific Sliders
@app.callback(
    Output('sector-levers-container', 'children'),
    Input('sector-dropdown', 'value'),
    State('models-store', 'data') # Use State since we only need to read it
)
def update_sectoral_levers(selected_sector, models_data):
    if not selected_sector or not models_data:
        return []

    features_map = models_data['features_map']
    sector_features = features_map[selected_sector]
    
    # Dynamically create a list of sliders
    dynamic_sliders = []
    
    # --- 🐟 Cod Levers ---
    if "Cod_Quota_kT" in sector_features:
        dynamic_sliders.extend([
            html.Label("Icelandic Cod Quota (% Change)"),
            dcc.Slider(id={'type': 'sector-slider', 'feature': 'Cod_Quota_kT'}, min=-20, max=20, step=1, value=0, marks={-20: '-20%', 20: '+20%'})
        ])
    if "Norwegian_Quota_kT" in sector_features:
        dynamic_sliders.extend([
            html.Label("Norwegian/Russian Quota (% Change)"),
            dcc.Slider(id={'type': 'sector-slider', 'feature': 'Norwegian_Quota_kT'}, min=-20, max=20, step=1, value=0, marks={-20: '-20%', 20: '+20%'})
        ])
        
    # --- ✈️ Tourism Levers ---
    if "Airline_Seat_Capacity" in sector_features:
        dynamic_sliders.extend([
            html.Label("Airline Seat Capacity (% Change)"),
            dcc.Slider(id={'type': 'sector-slider', 'feature': 'Airline_Seat_Capacity'}, min=-25, max=25, step=1, value=0, marks={-25: '-25%', 25: '+25%'})
        ])
    if "EU_Economic_Sentiment" in sector_features:
        dynamic_sliders.extend([
            html.Label("EU Economic Sentiment (% Change)"),
            dcc.Slider(id={'type': 'sector-slider', 'feature': 'EU_Economic_Sentiment'}, min=-10, max=10, step=1, value=0, marks={-10: '-10%', 10: '+10%'})
        ])
        
    # --- 🏭 Aluminum Levers ---
    if "LME_Aluminum_Price" in sector_features:
        dynamic_sliders.extend([
            html.Label("LME Aluminum Price (% Change)"),
            dcc.Slider(id={'type': 'sector-slider', 'feature': 'LME_Aluminum_Price'}, min=-30, max=30, step=1, value=0, marks={-30: '-30%', 30: '+30%'})
        ])
    if "Carbon_Credit_Price_EUR" in sector_features:
        dynamic_sliders.extend([
            html.Label("Carbon Credit Price (€/tonne)"),
            dcc.Slider(id={'type': 'sector-slider', 'feature': 'Carbon_Credit_Price_EUR'}, min=0, max=200, step=5, value=80, marks={0: '0€', 200: '200€'})
        ])

    return dynamic_sliders

# Callback 3: Run Simulation, Update Chart, and Update SHAP Plots
# This is the "main" callback that does all the work.
@app.callback(
    Output('main-forecast-chart', 'figure'),
    Output('simulation-data-store', 'data'),
    Output('shap-global-plot', 'src'),
    Input('sector-dropdown', 'value'),
    Input('cbank-rate-slider', 'value'),
    Input('oil-price-slider', 'value'),
    Input('wage-index-slider', 'value'),
    Input({'type': 'sector-slider', 'feature': ALL}, 'value'), # Pattern-matching
    State({'type': 'sector-slider', 'feature': ALL}, 'id'),   # Get IDs of dynamic sliders
    State('models-store', 'data')
)
def run_simulation_and_update_plots(
    selected_sector, rate_mod, oil_mod, wage_mod,
    dynamic_slider_values, dynamic_slider_ids, models_data):

    if not selected_sector or not models_data:
        return go.Figure(), {}, ""

    # --- 1. Rebuild Sectoral Inputs Dictionary ---
    sectoral_inputs = {}
    for i, slider_id_dict in enumerate(dynamic_slider_ids):
        feature_name = slider_id_dict['feature']
        feature_value = dynamic_slider_values[i]
        sectoral_inputs[feature_name] = feature_value

    # --- 2. Re-train simple model (Demo only) ---
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

    # --- 3. Run Simulation Logic ---
    future_dates = pd.date_range(start="2025-01-01", end="2026-12-31", freq='MS')
    n_future = len(future_dates)
    
    # This logic is copied from our Streamlit app's `run_simulation`
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

    # --- 4. Build X_future matrix ---
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

    # Build for "What-If"
    X_future_whatif = build_x_future(sectoral_inputs)
    predictions_whatif = model.predict(X_future_whatif)
    
    # Build for "Baseline"
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
    fig.update_layout(title=f"Forecast for {selected_sector.split(' ')[1]}", xaxis_title="Date", yaxis_title="Simulated Value")

    # --- 6. Generate Global SHAP Plot ---
    fig_global, ax_global = plt.subplots()
    shap.summary_plot(explainer(X_train), X_train, plot_type="bar", show=False)
    global_plot_src = convert_plt_to_base64(fig_global)

    # --- 7. Prepare Data for Local SHAP Plot ---
    # We store the simulation results in the dcc.Store
    # This way, the local SHAP callback only reads this data
    # and doesn't need to re-run the whole simulation.
    future_shap_values = explainer(X_future_whatif)
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]

    store_data = {
        "base_value": base_value,
        "shap_values": future_shap_values.values.tolist(), # Convert to list for JSON
        "x_future": X_future_whatif.to_json(),
        "features": sector_features,
        "predictions": predictions_whatif.tolist()
    }

    return fig, store_data, global_plot_src

# Callback 4: Update Local SHAP Dropdown
@app.callback(
    Output('shap-month-dropdown', 'options'),
    Output('shap-month-dropdown', 'value'),
    Input('simulation-data-store', 'data') # Triggered when simulation runs
)
def update_month_dropdown(store_data):
    if not store_data:
        return [], None
    
    future_dates = pd.date_range(start="2025-01-01", end="2026-12-31", freq='MS')
    options = [{'label': d.strftime('%B %Y'), 'value': i} for i, d in enumerate(future_dates)]
    
    return options, 0 # Default to the first month

# Callback 5: Update Local SHAP Plot
@app.callback(
    Output('shap-local-plot', 'src'),
    Input('shap-month-dropdown', 'value'), # Triggered by user selecting a month
    State('simulation-data-store', 'data') # Read the stored simulation data
)
def update_local_shap_plot(month_idx, store_data):
    if month_idx is None or not store_data:
        return ""
        
    # Read all data from the store
    base_value = store_data['base_value']
    shap_values = np.array(store_data['shap_values']) # Convert back to numpy
    X_future = pd.read_json(store_data['x_future'])
    features = store_data['features']
    
    # Get data for the selected month
    shap_values_for_month = shap_values[month_idx]
    data_for_month = X_future.iloc[month_idx]

    # Generate the waterfall plot
    fig_local, ax_local = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            base_values=base_value,
            values=shap_values_for_month,
            feature_names=features,
            data=data_for_month
        ),
        show=False
    )
    local_plot_src = convert_plt_to_base64(fig_local)
    
    return local_plot_src

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=8050)