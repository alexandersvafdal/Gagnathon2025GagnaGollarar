import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib  # To show where you would load models
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Icelandic Fish Export Forecast",
    page_icon="🐟",
    layout="wide"
)

# --- (SIMULATION) Model Training (In a real app, you'd skip this and just load files) ---
@st.cache_resource(show_spinner="Simulating model and data...")
def get_model_and_data():
    """
    This function simulates a pre-trained model, explainer, and data.
    In your real project, you would load these from disk (e.g., joblib.load).
    """
    # 1. Create Fake Historical Data
    dates = pd.date_range(start="2018-01-01", end="2024-12-31", freq='MS')
    n = len(dates)
    data = pd.DataFrame(index=dates)
    data['Cod_Quota_kT'] = np.random.uniform(250, 300, n).cumsum() / (np.arange(n) + 1)
    data['ISK_EUR_Rate'] = np.random.uniform(130, 150, n).cumsum() / (np.arange(n) + 1)
    data['SST_Anomaly'] = np.sin(np.arange(n) * 2 * np.pi / 12) + np.random.randn(n) * 0.2
    
    # Target Variable
    noise = np.random.randn(n) * 5
    data['Export_Value_B_ISK'] = (
        (data['Cod_Quota_kT'] / 10) + 
        (1 / data['ISK_EUR_Rate'] * 5000) + 
        (data['SST_Anomaly'] * 3) + 
        50 + noise
    )
    
    # 2. Create Fake Features (Lagging)
    features = ['Cod_Quota_kT', 'ISK_EUR_Rate', 'SST_Anomaly']
    X = data[features].shift(1).dropna()  # Use lagged features
    y = data['Export_Value_B_ISK'][1:]
    
    # 3. Train a dummy model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # 4. Create an explainer
    explainer = shap.TreeExplainer(model)
    
    # 5. Get base SHAP values
    shap_values = explainer(X)
    
    return model, explainer, data, X, features

# Load all the pre-trained objects
model, explainer, hist_data, X_train, features = get_model_and_data()

# --- App Layout ---

st.title("🐟 Icelandic Fish Export Forecast")
st.caption("An Explainable AI (XAI) Dashboard for Policy & Risk Simulation")

# --- Sidebar: What-If Controls ---
st.sidebar.header("What-If Scenario Controls")
st.sidebar.markdown("Adjust the baseline assumptions for the 2-year forecast (2025-2026).")

# Create future dates
future_dates = pd.date_range(start="2025-01-01", end="2026-12-31", freq='MS')
baseline_future_df = pd.DataFrame(index=future_dates)

# Get baseline values from the last known data point
baseline_quota = X_train['Cod_Quota_kT'].iloc[-1]
baseline_fx = X_train['ISK_EUR_Rate'].iloc[-1]
baseline_sst = X_train['SST_Anomaly'].iloc[-1] # This would normally be a forecast

# Create sliders for a % change from baseline
quota_mod = st.sidebar.slider(
    "Cod Quota (% Change)",
    min_value=-20.0, max_value=20.0, value=0.0, step=1.0,
    help="Adjust the % change from the baseline 2024 quota."
)

fx_mod = st.sidebar.slider(
    "ISK/EUR Rate (% Change)",
    min_value=-20.0, max_value=20.0, value=0.0, step=1.0,
    help="Adjust the % change from the baseline 2024 FX rate. (Lower = Stronger ISK)"
)

# --- Backend: Generate Forecast ---
def generate_forecast(quota_mod, fx_mod):
    """
    Generates a new forecast based on user inputs.
    THIS IS THE CORE OF THE "ON-THE-FLY" INTERACTIVITY.
    It's FAST because it's just calling model.predict() and explainer().
    """
    # 1. Create the future feature matrix (X_future)
    X_future = pd.DataFrame(index=future_dates, columns=features)
    
    # Apply user modifications
    X_future['Cod_Quota_kT'] = baseline_quota * (1 + quota_mod / 100)
    X_future['ISK_EUR_Rate'] = baseline_fx * (1 + fx_mod / 100)
    X_future['SST_Anomaly'] = baseline_sst # Keep it simple, no slider for this one
    
    # 2. Get predictions (FAST!)
    predictions = model.predict(X_future)
    forecast_df = pd.DataFrame({'Export_Value_B_ISK': predictions}, index=future_dates)
    
    # 3. Get SHAP values for the new forecast (Also FAST!)
    future_shap_values = explainer(X_future)
    
    return forecast_df, future_shap_values, X_future

# Run the forecast calculation
forecast_df, future_shap_values, X_future = generate_forecast(quota_mod, fx_mod)


# --- Main Page: Tabs ---
tab1, tab2 = st.tabs(["📈 Forecast & Scenario", "🔍 Explainability (SHAP)"])

with tab1:
    st.header("24-Month Export Value Forecast")
    
    # Create Plotly figure
    fig = go.Figure()
    
    # 1. Add historical data
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['Export_Value_B_ISK'],
        mode='lines',
        name='Historical Export Value',
        line=dict(color='gray')
    ))
    
    # 2. Add forecast data (from the sliders)
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['Export_Value_B_ISK'],
        mode='lines+markers',
        name='Forecast (What-If Scenario)',
        line=dict(color='red', dash='dot', width=3)
    ))
    
    # (Optional) Add a dummy baseline to show the change
    baseline_forecast, _, _ = generate_forecast(0.0, 0.0) # Run with 0% change
    fig.add_trace(go.Scatter(
        x=baseline_forecast.index,
        y=baseline_forecast['Export_Value_B_ISK'],
        mode='lines',
        name='Baseline Forecast',
        line=dict(color='blue', dash='dash')
    ))

    fig.update_layout(
        title="Historical Data vs. 'What-If' Forecast",
        xaxis_title="Date",
        yaxis_title="Export Value (Billion ISK)",
        legend_title="Series"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Understanding the Forecast Drivers")
    st.markdown("Powered by SHAP (SHapley Additive exPlanations)")

    st.subheader("Global Feature Importance")
    st.markdown("Which features have the biggest impact on the forecast *overall*?")
    
    # --- Global Importance Plot (Summary) ---
    # We use the SHAP values from the *training* data (X) for the global plot
    shap.summary_plot(explainer(X_train), X_train, plot_type="bar", show=False)
    st.pyplot(plt.gcf(), clear_figure=True) # Display the plot and clear it
    
    st.divider()
    
    st.subheader("Local Forecast Explanation")
    st.markdown("Why did the model predict a specific value for a *single month*?")

    # --- Local Explanation ---
    
    # 1. Create a list of formatted date strings for the dropdown
    formatted_dates = [d.strftime('%B %Y') for d in future_dates]
    
    # 2. Create the dropdown (selectbox)
    selected_date_str = st.selectbox(
        "Select a forecast month to explain:",
        options=formatted_dates
    )
    
    # 3. Find the integer index of the date the user selected
    month_idx = formatted_dates.index(selected_date_str)
    
    st.markdown(f"**Explanation for {selected_date_str}:**")
    
    # --- Local Plot ---
    # Get the base value (average prediction)
    base_value = explainer.expected_value
    
    # Get the SHAP values for the selected month
    shap_values_for_month = future_shap_values.values[month_idx]
    
    # Get the feature values for the selected month
    data_for_month = X_future.iloc[month_idx]
    
    # Generate the waterfall plot
    shap.waterfall_plot(
        shap.Explanation(
            base_values=base_value[0], # <-- Can also add [0] here, good practice
            values=shap_values_for_month,
            feature_names=features,
            data=data_for_month
        ),
        show=False
    )
    st.pyplot(plt.gcf(), clear_figure=True) # Display the plot
    
    st.markdown(f"""
    **How to read this plot:**
    * `E[f(x)]` is the model's average "base" prediction (e.g., **{base_value[0]:.2f}** Billion ISK). 
    * Red bars (like Cod Quota) are features **pushing the forecast UP** for this month.
    * Blue bars (like ISK/EUR Rate) are features **pulling the forecast DOWN**.
    * `f(x)` is the final forecast for this month (e.g., **{forecast_df['Export_Value_B_ISK'].iloc[month_idx]:.2f}** Billion ISK).
    """)