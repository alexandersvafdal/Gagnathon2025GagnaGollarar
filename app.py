import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Icelandic Economic Simulator",
    page_icon="🌍",
    layout="wide"
)

# --- (SIMULATION) Model & Data Generation ---
@st.cache_resource(show_spinner="Simulating Iceland's Economy (Models & Data)...")
def get_all_models_and_data():
    """
    **MODIFIED**: This function now builds dummy data with
    *strong, hard-coded relationships* so the dummy models
    become highly sensitive to the slider inputs.
    """
    SECTORS = ["🐟 Cod Exports", "✈️ Tourism", "🏭 Aluminum"]
    
    models = {}
    explainers = {}
    historical_data = {}
    training_data = {}
    
    # Define the features for each model
    features_map = {
        "🐟 Cod Exports": ["Cod_Quota_kT", "ISK_EUR_Rate", "Marine_Fuel_Price"],
        "✈️ Tourism": ["ISK_EUR_Rate", "EU_Economic_Sentiment", "Jet_Fuel_Price"],
        "🏭 Aluminum": ["LME_Aluminum_Price", "Global_Industrial_Production", "Electricity_Cost"]
    }
    
    # --- Generate a model for each sector ---
    for sector in SECTORS:
        features = features_map[sector]
        
        # 1. Create Dates
        dates = pd.date_range(start="2018-01-01", end="2024-12-31", freq='MS')
        n = len(dates)
        data = pd.DataFrame(index=dates)
        
        # 2. Create Dummy Features with structure
        X = pd.DataFrame(index=data.index[1:], columns=features)
        base_time = np.arange(n-1)
        seasonal = np.sin(base_time * 2 * np.pi / 12) * 50
        
        # Create different-looking dummy features
        X[features[0]] = 100 + base_time * 0.1 + seasonal + np.random.randn(n-1)*5
        X[features[1]] = 140 + (np.random.randn(n-1)*1.5).cumsum()
        X[features[2]] = 80 + seasonal * 0.5 + np.random.randn(n-1)*2

        # 3. **IMPORTANT**: Create Dummy Target Variable with strong relationships
        y = 50 + np.random.randn(n-1)*2 # Base value + noise
        
        # Add the strong relationships
        y += (X[features[0]] - 100) * 2.5  # Feature 0 is very important (positive)
        y -= (X[features[1]] - 140) * 1.5  # Feature 1 is important (negative)
        y -= (X[features[2]] - 80) * 0.5   # Feature 2 is less important (negative)

        # Store for historical plot
        data['Target_Value'] = np.nan
        data.iloc[1:, data.columns.get_loc('Target_Value')] = y

        # 4. Train a dummy model
        # This model will now *learn* the strong relationships we just created
        model = RandomForestRegressor(n_estimators=30, random_state=42)
        model.fit(X, y)
        
        # 5. Create an explainer
        explainer = shap.TreeExplainer(model)
        
        # 6. Store in our dictionaries
        models[sector] = model
        explainers[sector] = explainer
        historical_data[sector] = data
        training_data[sector] = X
    
    return models, explainers, historical_data, training_data, features_map, SECTORS

# --- Load all pre-trained objects ---
models, explainers, hist_data_map, X_train_map, features_map, SECTORS = get_all_models_and_data()

# --- App Layout ---
st.title("🌍 Icelandic Macro-Economic Simulator")
st.caption("An Explainable AI (XAI) Dashboard for Policy & Risk Simulation")

# --- Sidebar: Product Selection ---
st.sidebar.header("1. Select Sector")
selected_sector = st.sidebar.selectbox(
    "Select Sector to Forecast:",
    options=SECTORS
)

# --- Get Sector-Specific Objects ---
model = models[selected_sector]
explainer = explainers[selected_sector]
hist_data = hist_data_map[selected_sector]
X_train = X_train_map[selected_sector]
sector_features = features_map[selected_sector]

# --- Sidebar: Macro Levers (GLOBAL) ---
st.sidebar.header("2. 🌍 Macro-Economic Levers")
st.sidebar.markdown("These levers affect *all* sectors in the simulation.")

cbank_rate_mod = st.sidebar.slider(
    "Central Bank Rate (Rel. %)",
    -2.0, 2.0, 0.0, 0.25,
    help="Change in the Central Bank's main interest rate (e.g., +0.5%)."
)

oil_price_mod = st.sidebar.slider(
    "Brent Oil Price (% Change)",
    -30.0, 50.0, 0.0, 5.0,
    help="Assumed % change in global oil prices."
)

wage_index_mod = st.sidebar.slider(
    "Wage Index (% Change)",
    -5.0, 10.0, 0.0, 1.0,
    help="Assumed % change in the national wage index."
)
st.sidebar.markdown("---")

# --- Sidebar: Sectoral Levers (DYNAMIC) ---
st.sidebar.header(f"3. {selected_sector.split(' ')[0]} Sectoral Levers")
st.sidebar.markdown("These levers *only* affect the selected sector.")

# This section is now dynamic based on the selected sector
sectoral_inputs = {}
if "Cod_Quota_kT" in sector_features:
    sectoral_inputs["Cod_Quota_kT"] = st.sidebar.slider(
        "Cod Quota (% Change)", -20.0, 20.0, 0.0, 1.0
    )
if "LME_Aluminum_Price" in sector_features:
    sectoral_inputs["LME_Aluminum_Price"] = st.sidebar.slider(
        "LME Aluminum Price (% Change)", -30.0, 30.0, 0.0, 1.0
    )
if "EU_Economic_Sentiment" in sector_features:
    sectoral_inputs["EU_Economic_Sentiment"] = st.sidebar.slider(
        "EU Economic Sentiment (% Change)", -10.0, 10.0, 0.0, 1.0
    )

st.sidebar.info("Note: Macro Levers (like Interest Rates) affect Sectoral drivers (like the ISK/EUR Rate). The SHAP plot will show the impact of the *driver* (ISK/EUR Rate), not the *lever* (Interest Rate).")

# --- Backend: System Simulation ---
future_dates = pd.date_range(start="2025-01-01", end="2026-12-31", freq='MS')
n_future = len(future_dates)

def run_simulation(rate_mod, oil_mod, wage_mod, sector_inputs):
    """
    This function takes the user's slider inputs and simulates
    the future feature values (X_future) to be fed into the model.
    """
    
    # --- 1. Simulate Intermediate Drivers (based on Macro Levers) ---
    
    # SIMULATION: Higher interest rates -> stronger ISK (lower rate)
    # SIMULATION: Higher wages (inflation) -> weaker ISK (higher rate)
    fx_rate_change = (wage_mod * 0.8) - (rate_mod * 2.0) # <-- Increased sensitivity
    isk_eur_rate_future = 145 * (1 + fx_rate_change / 100)
    
    # SIMULATION: Higher oil price -> higher fuel costs
    marine_fuel_future = 100 * (1 + (oil_mod * 1.5) / 100) # <-- Amplified effect
    jet_fuel_future = 120 * (1 + (oil_mod * 1.2) / 100)   # <-- Amplified effect
    
    # SIMULATION: Other drivers
    eu_sentiment_future = 100 * (1 + (wage_mod / 100 * 0.5) - (oil_mod / 100 * 0.2))
    global_prod_future = 100 * (1 + (wage_mod / 100 * 0.3) - (oil_mod / 100 * 0.3))
    electricity_cost_future = 50 * (1 - rate_mod / 100)
    
    # Create a dictionary of all possible future drivers
    future_driver_values = {
        "ISK_EUR_Rate": np.full(n_future, isk_eur_rate_future),
        "Marine_Fuel_Price": np.full(n_future, marine_fuel_future),
        "Jet_Fuel_Price": np.full(n_future, jet_fuel_future),
        "EU_Economic_Sentiment": np.full(n_future, eu_sentiment_future),
        "Global_Industrial_Production": np.full(n_future, global_prod_future),
        "Electricity_Cost": np.full(n_future, electricity_cost_future),
        
        # Add a seasonal component to one driver for a non-flat line
        "seasonal_driver": np.sin(np.arange(n_future) * 2 * np.pi / 12) * 50
    }

    # --- 2. Build X_future matrix for the SELECTED sector ---
    X_future = pd.DataFrame(index=future_dates, columns=sector_features)
    
    for feature in sector_features:
        if feature in sector_inputs:
            # This is a direct sectoral lever (e.g., Cod Quota)
            base_val = 110 # Dummy base
            mod_val = sector_inputs[feature]
            X_future[feature] = base_val * (1 + mod_val / 100) + future_driver_values["seasonal_driver"] # Add season
        
        elif feature in future_driver_values:
            # This is an intermediate driver (e.g., ISK_EUR_Rate)
            X_future[feature] = future_driver_values[feature]
            
        else:
            # Fallback for any missing dummy data
            X_future[feature] = 100 # Dummy base
            
    # --- 3. Run Model & Explainer ---
    # The model (which learned strong relationships) will
    # now be highly responsive to these input changes.
    predictions = model.predict(X_future)
    forecast_df = pd.DataFrame({'Target_Value': predictions}, index=future_dates)
    future_shap_values = explainer(X_future)
    
    return forecast_df, future_shap_values, X_future

# --- Run the Simulation ---
# Run for the "what-if" scenario
forecast_df, future_shap_values, X_future = run_simulation(
    cbank_rate_mod, oil_price_mod, wage_index_mod, sectoral_inputs
)
# Run for the "baseline" (all mods at 0)
baseline_sector_inputs = {k: 0.0 for k in sectoral_inputs}
baseline_forecast_df, _, _ = run_simulation(
    0.0, 0.0, 0.0, baseline_sector_inputs
)


# --- Main Page: Tabs ---
tab1, tab2 = st.tabs(["📈 Forecast & Scenario", "🔍 Explainability (SHAP)"])

with tab1:
    st.header(f"24-Month Forecast for {selected_sector}")
    
    fig = go.Figure()
    
    # 1. Add historical data
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['Target_Value'],
        mode='lines',
        name=f'Historical ({selected_sector.split(" ")[1]})',
        line=dict(color='gray', width=2, dash='dot') # <-- COLOR CHANGE
    ))
    
    # 3. Add "Baseline" forecast
    fig.add_trace(go.Scatter(
        x=baseline_forecast_df.index,
        y=baseline_forecast_df['Target_Value'],
        mode='lines',
        name='Baseline Forecast',
        line=dict(color='#007BFF', dash='dash', width=3) # <-- COLOR CHANGE (Blue)
    ))

    # 2. Add "What-If" forecast (plotted on top)
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['Target_Value'],
        mode='lines+markers',
        name='Forecast (What-If Scenario)',
        line=dict(color='#FF006E', width=4), # <-- COLOR CHANGE (Vibrant Red/Pink)
        marker=dict(size=5, color='#FF006E')
    ))

    fig.update_layout(
        title=f"Historical vs. 'What-If' Forecast for {selected_sector.split(' ')[1]}",
        xaxis_title="Date",
        yaxis_title="Simulated Value (e.g., Billion ISK or Arrivals)",
        legend_title="Series"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header(f"Understanding the Forecast Drivers for {selected_sector}")
    st.markdown("Powered by SHAP (SHapley Additive exPlanations)")

    st.subheader("Global Feature Importance")
    st.markdown(f"Which features have the biggest impact on the **{selected_sector}** model *overall*?")
    
    shap.summary_plot(explainer(X_train), X_train, plot_type="bar", show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    
    st.divider()
    
    st.subheader("Local Forecast Explanation")
    st.markdown("Why did the model predict a specific value for a *single month*?")

    # --- Local Explanation Dropdown ---
    formatted_dates = [d.strftime('%B %Y') for d in future_dates]
    selected_date_str = st.selectbox(
        "Select a forecast month to explain:",
        options=formatted_dates,
        key=f"selectbox_{selected_sector}" # Dynamic key to prevent widget errors
    )
    month_idx = formatted_dates.index(selected_date_str)
    
    st.markdown(f"**Explanation for {selected_date_str}:**")
    
    # --- Local Plot ---
    base_value = explainer.expected_value
    shap_values_for_month = future_shap_values.values[month_idx]
    data_for_month = X_future.iloc[month_idx]
    
    # Handle base_value being an array
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]

    shap.waterfall_plot(
        shap.Explanation(
            base_values=base_value,
            values=shap_values_for_month,
            feature_names=sector_features,
            data=data_for_month
        ),
        show=False
    )
    st.pyplot(plt.gcf(), clear_figure=True)
    
    st.markdown(f"""
    **How to read this plot:**
    * `E[f(x)]` is the model's average "base" prediction (e.g., **{base_value:.2f}**).
    * Red bars are features **pushing the forecast UP** for this month.
    * Blue bars are features **pulling the forecast DOWN**.
    * `f(x)` is the final forecast for this month (e.g., **{forecast_df['Target_Value'].iloc[month_idx]:.2f}**).
    """)