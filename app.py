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
    page_title="Icelandic Fish Export Forecast",
    page_icon="🐟",
    layout="wide"
)

# --- (SIMULATION) Model Training ---
@st.cache_resource(show_spinner="Simulating all product models...")
def get_model_and_data():
    """
    **NEW**: This function now simulates data and trains a separate
    model for multiple products.
    """
    PRODUCTS = ["Cod", "Salmon", "Haddock"]
    
    # We will store all our models, data, etc., in dictionaries
    # keyed by the product name.
    models = {}
    explainers = {}
    historical_data = {}
    training_data = {}
    features = ['Main_Quota_kT', 'ISK_EUR_Rate', 'SST_Anomaly']
    
    for i, product in enumerate(PRODUCTS):
        # 1. Create Dates
        dates = pd.date_range(start="2018-01-01", end="2024-12-31", freq='MS')
        n = len(dates)
        
        # 2. Create Features
        data = pd.DataFrame(index=dates)
        
        # We'll make the dummy data different for each product
        base_quota = 250 - i * 50 # Cod=250, Salmon=200, Haddock=150
        base_fx = 140 + i * 5     # Cod=140, Salmon=145, Haddock=150
        seasonal_strength = (i + 1) * 2 # Cod=2, Salmon=4, Haddock=6
        
        data['Main_Quota_kT'] = base_quota + (np.arange(n) / 5) + np.random.randn(n) * 2
        data['ISK_EUR_Rate'] = base_fx + (np.random.randn(n) * 1.5).cumsum() / 5
        data['SST_Anomaly'] = np.sin(np.arange(n) * 2 * np.pi / 12) * seasonal_strength + np.random.randn(n) * 0.5
        
        # 3. Create Target Variable (Export Value)
        base_value = 50 - i * 15 # Cod=50, Salmon=35, Haddock=20
        data['Export_Value_B_ISK'] = (
            base_value + 
            (data['Main_Quota_kT'] / 10) - 
            (data['ISK_EUR_Rate'] / 20) + 
            (data['SST_Anomaly'] * 2) +  # Seasonal driver
            np.random.randn(n) * 2      # Noise
        )
        
        # 4. Standard ML setup
        X = data[features].shift(1).dropna()
        y = data['Export_Value_B_ISK'][1:]
        
        # 5. Train a product-specific model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # 6. Create a product-specific explainer
        explainer = shap.TreeExplainer(model)
        
        # 7. Store in our dictionaries
        models[product] = model
        explainers[product] = explainer
        historical_data[product] = data
        training_data[product] = X
    
    return models, explainers, historical_data, training_data, features, PRODUCTS

# Load all the pre-trained objects
models, explainers, historical_data, training_data, features, PRODUCTS = get_model_and_data()

# --- App Layout ---
st.title("🐟 Icelandic Fish Export Forecast")
st.caption("An Explainable AI (XAI) Dashboard for Policy & Risk Simulation")

# --- Sidebar: Product Selection ---
st.sidebar.header("Product Selection")
selected_product = st.sidebar.selectbox(
    "Select Product to Forecast:",
    options=PRODUCTS
)

# --- Sidebar: What-If Controls ---
st.sidebar.header("What-If Scenario Controls")
st.sidebar.markdown(f"Adjust the assumptions for **{selected_product}**.")

# --- Get Product-Specific Objects ---
# NOW, all main variables (model, data, etc.) are for the
# product the user selected in the sidebar.
model = models[selected_product]
explainer = explainers[selected_product]
hist_data = historical_data[selected_product]
X_train = training_data[selected_product]
hist_n = len(hist_data) # Get length of historical data

# Create future dates
future_dates = pd.date_range(start="2025-01-01", end="2026-12-31", freq='MS')

# Get baseline values from the last known data point *for this product*
baseline_quota = X_train[features[0]].iloc[-1]
baseline_fx = X_train[features[1]].iloc[-1]

# Create sliders
quota_mod = st.sidebar.slider(
    "Main Quota (% Change)", # Renamed from "Cod Quota"
    min_value=-20.0, max_value=20.0, value=0.0, step=1.0,
    help=f"Adjust the % change from the baseline 2024 quota for {selected_product}."
)

fx_mod = st.sidebar.slider(
    "ISK_EUR Rate (% Change)",
    min_value=-20.0, max_value=20.0, value=0.0, step=1.0,
    help="Adjust the % change from the baseline 2024 FX rate."
)

# --- Backend: Generate Forecast ---
def generate_future_features(quota_mod, fx_mod, base_quota, base_fx, hist_n):
    """
    Generates the X_future matrix based on user inputs.
    """
    X_future = pd.DataFrame(index=future_dates, columns=features)
    
    # Apply user modifications
    X_future[features[0]] = base_quota * (1 + quota_mod / 100)
    X_future[features[1]] = base_fx * (1 + fx_mod / 100)
    
    # Create the future seasonal wave
    future_n = len(future_dates)
    future_time_index = np.arange(hist_n, hist_n + future_n)
    
    # We need to know which product to get the right seasonal strength
    product_index = PRODUCTS.index(selected_product)
    seasonal_strength = (product_index + 1) * 2
    
    future_sine_wave = np.sin(future_time_index * 2 * np.pi / 12) * seasonal_strength
    
    X_future[features[2]] = future_sine_wave
    
    return X_future

# --- Run Calculation ---
# 1. Generate the future features
X_future = generate_future_features(quota_mod, fx_mod, baseline_quota, baseline_fx, hist_n)

# 2. Use the product-specific model to predict
predictions = model.predict(X_future)
forecast_df = pd.DataFrame({'Export_Value_B_ISK': predictions}, index=future_dates)

# 3. Use the product-specific explainer
future_shap_values = explainer(X_future)


# --- Main Page: Tabs ---
tab1, tab2 = st.tabs(["📈 Forecast & Scenario", "🔍 Explainability (SHAP)"])

with tab1:
    # Title is now dynamic
    st.header(f"24-Month Export Value Forecast for {selected_product}")
    
    # Create Plotly figure
    fig = go.Figure()
    
    # 1. Add historical data (for the selected product)
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['Export_Value_B_ISK'],
        mode='lines',
        name=f'Historical ({selected_product})',
        line=dict(color='silver', width=2)
    ))
    
    # 2. Add forecast data (from the sliders)
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['Export_Value_B_ISK'],
        mode='lines+markers',
        name='Forecast (What-If Scenario)',
        line=dict(color='#00FFFF', dash='dot', width=4),
        marker=dict(size=5)
    ))
    
    # 3. Add baseline forecast
    # We need to re-generate the baseline features
    baseline_X_future = generate_future_features(0.0, 0.0, baseline_quota, baseline_fx, hist_n)
    baseline_predictions = model.predict(baseline_X_future)
    baseline_forecast_df = pd.DataFrame({'Export_Value_B_ISK': baseline_predictions}, index=future_dates)
    
    fig.add_trace(go.Scatter(
        x=baseline_forecast_df.index,
        y=baseline_forecast_df['Export_Value_B_ISK'],
        mode='lines',
        name='Baseline Forecast',
        line=dict(color='#FF8C00', dash='dash', width=3)
    ))

    fig.update_layout(
        title=f"Historical vs. 'What-If' Forecast for {selected_product}",
        xaxis_title="Date",
        yaxis_title="Export Value (Billion ISK)",
        legend_title="Series"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header(f"Understanding the Forecast Drivers for {selected_product}")
    st.markdown("Powered by SHAP (SHapley Additive exPlanations)")

    st.subheader("Global Feature Importance")
    st.markdown(f"Which features have the biggest impact on the **{selected_product}** forecast *overall*?")
    
    # --- Global Importance Plot (Summary) ---
    # This now uses the X_train for the selected product
    shap.summary_plot(explainer(X_train), X_train, plot_type="bar", show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    
    st.divider()
    
    st.subheader("Local Forecast Explanation")
    st.markdown("Why did the model predict a specific value for a *single month*?")

    # --- Local Explanation ---
    
    # 1. Create a list of formatted date strings
    formatted_dates = [d.strftime('%B %Y') for d in future_dates]
    
    # 2. Create the dropdown (selectbox)
    selected_date_str = st.selectbox(
        "Select a forecast month to explain:",
        options=formatted_dates,
        key=f"selectbox_{selected_product}" # Add a dynamic key
    )
    
    # 3. Find the integer index
    month_idx = formatted_dates.index(selected_date_str)
    
    st.markdown(f"**Explanation for {selected_date_str}:**")
    
    # --- Local Plot ---
    # These variables are already product-specific
    base_value = explainer.expected_value
    shap_values_for_month = future_shap_values.values[month_idx]
    data_for_month = X_future.iloc[month_idx]
    
    # Generate the waterfall plot
    shap.waterfall_plot(
        shap.Explanation(
            base_values=base_value[0],
            values=shap_values_for_month,
            feature_names=features,
            data=data_for_month
        ),
        show=False
    )
    st.pyplot(plt.gcf(), clear_figure=True)
    
    st.markdown(f"""
    **How to read this plot:**
    * `E[f(x)]` is the model's average "base" prediction (e.g., **{base_value[0]:.2f}** Billion ISK).
    * Red bars are features **pushing the forecast UP** for this month.
    * Blue bars are features **pulling the forecast DOWN**.
    * `f(x)` is the final forecast for this month (e.g., **{forecast_df['Export_Value_B_ISK'].iloc[month_idx]:.2f}** Billion ISK).
    """)