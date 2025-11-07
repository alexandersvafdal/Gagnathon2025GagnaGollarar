import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import itertools
import shap
import json
import matplotlib
matplotlib.use('Agg') # Use non-GUI backend
import matplotlib.pyplot as plt # <-- FIX: This line was missing
import io
import base64

# --- Matplotlib (SHAP) to HTML Image Converter ---
def convert_plt_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig) # Close the plot to save memory
    data = base64.b64encode(buf.getbuffer()).decode("utf8")
    return "data:image/png;base64," + data

# --- Converter for transparent (dark) plots ---
def convert_plt_to_base64_transparent(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', transparent=True)
    plt.close(fig)
    data = base64.b64encode(buf.getbuffer()).decode("utf8")
    return "data:image/png;base64," + data

# --- 1. CONFIGURATION ---
SECTORS = ["🐟 Cod Exports", "✈️ Tourism", "🏭 Aluminum"]

# These features MUST match the columns we create in `load_all_data`
FEATURES_MAP = {
    "🐟 Cod Exports": [
        "Cod_Quota_kT", # Dummy for now
        "Norwegian_Quota_kT", # Dummy for now
        "ISK_EUR_Rate_lag1", 
        "Inflation_Annual_Pct_lag1"
    ],
    "✈️ Tourism": [
        "Airline_Seat_Capacity", # Dummy for now
        "ISK_EUR_Rate_lag1", 
        "EU_Economic_Sentiment", # Dummy for now
        "Inflation_Annual_Pct_lag1"
    ],
    "🏭 Aluminum": [
        "LME_Aluminum_Price", # Dummy for now
        "Carbon_Credit_Price_EUR", # Dummy for now
        "Global_Industrial_Production", # Dummy for now
        "ISK_EUR_Rate_lag1"
    ]
}

# --- DEFINE YOUR SCENARIOS ---
# These will become the RadioItems in the app
SCENARIO_LEVELS = {
    'rate_mod': [-1.0, 0.0, 1.0],         # Central Bank Rate
    'oil_mod': [-20.0, 0.0, 20.0],       # Oil Price
    'wage_mod': [-5.0, 0.0, 5.0],         # Wage Index
    
    # Generic sectoral levels
    'sector_mod_1': [-20.0, 0.0, 20.0],  # e.g., Cod Quota, Seat Capacity
    'sector_mod_2': [-20.0, 0.0, 20.0]   # e.g., Norwegian Quota, LME Price
}

# --- 2. REAL DATA LOADING & CLEANING ---

def load_fish_exports():
    """
    --- UPDATED: This function now loads your real Parquet file ---
    Loads and aggregates fish exports from the main Parquet file.
    Groups by month and sums all fish-related branches.
    """
    try:
        df = pd.read_parquet("data/Exports-by-branches-of-processing-and-countries-2015-2025.parquet")
        
        # 1. Filter for 'Fob value' only
        df_fob = df[df['Unit'] == 'Fob value'].copy()

        # 2. Convert "Month" (e.g., "2020M01") to datetime
        df_fob['date'] = pd.to_datetime(df_fob['Month'], format='%YM%m')
        
        # --- FIX: Convert the zero-padded string column to a numeric float ---
        df_fob['DATA'] = pd.to_numeric(df_fob['DATA'], errors='coerce').fillna(0)

        # 4. Group by date and sum all branches/countries to get TOTAL fish export value
        df_agg = df_fob.groupby('date')['DATA'].sum().reset_index()
        
        # 5. Set index and resample to ensure clean Monthly Start index
        df_agg = df_agg.set_index('date').resample('MS').sum()
        
        # 6. Rename 'DATA' column to the name our model expects
        df_agg = df_agg.rename(columns={'DATA': 'Fish_Export_Value_FOB'})
        
        print(f"Loaded {len(df_agg)} months of Fish Export data.")
        return df_agg[['Fish_Export_Value_FOB']]
        
    except Exception as e:
        print(f"Fish export file not found or error: {e}. Using dummy data.")
        return pd.DataFrame(index=pd.date_range(start="2018-01-01", end="2024-12-31", freq='MS'), data={'Fish_Export_Value_FOB': np.random.rand(84) * 1000 + 50000})

def load_tourism_exports():
    """
    Loads and processes 'External trade in services' to get Tourism (Travel) data.
    """
    try:
        df = pd.read_csv("data/External trade in services, monthly 2018-2025.csv", sep=';', header=2)
        df = df.rename(columns={df.columns[0]: 'Service'})
        df_travel = df[df['Service'].str.contains('Travel', case=False, na=False)].copy()
        df_long = df_travel.melt(id_vars=['Service'], var_name='month_str', value_name='Tourism_Export_Value')
        df_long['date'] = pd.to_datetime(df_long['month_str'], format='%YM%m', errors='coerce')
        df_long = df_long.dropna(subset=['date']) # Drop columns that aren't dates
        # Clean numeric values (they have '.' for thousands and ',' for decimal)
        df_long['Tourism_Export_Value'] = pd.to_numeric(df_long['Tourism_Export_Value'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.'), errors='coerce')
        df_agg = df_long.groupby('date')[['Tourism_Export_Value']].sum().resample('MS').sum()
        print(f"Loaded {len(df_agg)} months of Tourism (Travel) data.")
        return df_agg
    except Exception as e:
        print(f"Tourism export file not found or error: {e}. Using dummy data.")
        return pd.DataFrame(index=pd.date_range(start="2018-01-01", end="2024-12-31", freq='MS'), data={'Tourism_Export_Value': np.random.rand(84) * 1000 + 20000})

def load_macro_features():
    """
    Loads, cleans, and merges Inflation and Exchange Rate data.
    """
    try:
        # Load Inflation (already monthly)
        df_inf = pd.read_csv("data/Inflation-Consumer price index.csv")
        # Handle quoted date string, e.g., "2015M01"
        df_inf['date'] = pd.to_datetime(df_inf['Month'].str.strip('"'), format='%YM%m')
        df_inf = df_inf.set_index('date')
        df_inf = df_inf[['Consumer price index Annual change, %']].rename(columns={'Consumer price index Annual change, %': 'Inflation_Annual_Pct'})
        print(f"Loaded {len(df_inf)} months of Inflation data.")
        
        # Load Exchange Rates (daily)
        df_fx = pd.read_csv("data/Exchange-rates_2015-2025.csv", sep=';', decimal=',', usecols=['Dagsetning', 'Evra EUR miðgengi', 'Bandaríkjadalur USD miðgengi'])
        df_fx['date'] = pd.to_datetime(df_fx['Dagsetning'], format='%d.%m.%Y')
        df_fx = df_fx.set_index('date')
        
        # Resample daily data to Monthly Start ('MS') average
        df_fx_monthly = df_fx[['Evra EUR miðgengi', 'Bandaríkjadalur USD miðgengi']].resample('MS').mean()
        df_fx_monthly = df_fx_monthly.rename(columns={'Evra EUR miðgengi': 'ISK_EUR_Rate', 'Bandaríkjadalur USD miðgengi': 'ISK_USD_Rate'})
        print(f"Loaded and resampled {len(df_fx_monthly)} months of FX data.")
        
        # Join them together
        df_macro = df_inf.join(df_fx_monthly, how='outer')
        return df_macro
        
    except Exception as e:
        print(f"Macro data files not found or error: {e}. Using dummy data.")
        dates = pd.date_range(start="2018-01-01", end="2024-12-31", freq='MS')
        return pd.DataFrame(index=dates, data={
            'Inflation_Annual_Pct': np.random.rand(len(dates)) * 5,
            'ISK_EUR_Rate': np.random.rand(len(dates)) * 20 + 130
        })

def load_all_data():
    """
    Loads all datasets and merges them into one master monthly DataFrame.
    """
    df_fish = load_fish_exports()
    df_tourism = load_tourism_exports()
    df_macro = load_macro_features()
    
    # For Aluminum, we'll use a dummy target for now
    dates = pd.date_range(start="2018-01-01", end="2024-12-31", freq='MS')
    df_aluminum = pd.DataFrame(index=dates, data={'Aluminum_Export_Value': np.random.rand(len(dates)) * 1000 + 30000})

    # Join all data sources
    df_master = df_macro.join(df_fish, how='outer')
    df_master = df_master.join(df_tourism, how='outer')
    df_master = df_master.join(df_aluminum, how='outer')
    
    # --- Create Lagged Features ---
    # This is how we solve the "knowing the future" problem.
    df_master['ISK_EUR_Rate_lag1'] = df_master['ISK_EUR_Rate'].shift(1)
    df_master['Inflation_Annual_Pct_lag1'] = df_master['Inflation_Annual_Pct'].shift(1)
    
    # Add dummy features for the ones we don't have real data for
    df_master['Cod_Quota_kT'] = 150 + np.sin(df_master.index.month * 2 * np.pi / 12) * 5
    df_master['Norwegian_Quota_kT'] = 200 + np.sin(df_master.index.month * 2 * np.pi / 12) * 10
    df_master['Airline_Seat_Capacity'] = 500000 + np.sin(df_master.index.month * 2 * np.pi / 12) * 100000
    df_master['EU_Economic_Sentiment'] = 100 + np.random.randn(len(df_master)) * 2
    df_master['LME_Aluminum_Price'] = 2000 + np.random.randn(len(df_master)) * 100
    df_master['Carbon_Credit_Price_EUR'] = 80 + np.random.randn(len(df_master)) * 5
    df_master['Global_Industrial_Production'] = 100 + np.random.randn(len(df_master)) * 2
    
    # Drop any rows with missing data (especially the first row from lagging)
    df_master = df_master.dropna(subset=['ISK_EUR_Rate_lag1', 'Inflation_Annual_Pct_lag1'])
    
    print("\nMaster DataFrame created:")
    print(df_master.head())
    
    return df_master

def get_training_data(sector, master_df):
    """
    Selects the correct Y (target) and X (features) for a given sector
    from the master DataFrame.
    """
    features = FEATURES_MAP[sector]
    
    if sector == "🐟 Cod Exports":
        y_col = "Fish_Export_Value_FOB"
    elif sector == "✈️ Tourism":
        y_col = "Tourism_Export_Value"
    elif sector == "🏭 Aluminum":
        y_col = "Aluminum_Export_Value"
    else:
        # Fallback for sectors without real data yet
        X, y, hist = get_dummy_data(sector)
        return X, y, hist

    # Drop rows where target or features are missing
    sector_df = master_df.dropna(subset=[y_col] + features)

    if sector_df.empty:
        print(f"Warning: No valid training data found for {sector}. Using dummy data.")
        X, y, hist = get_dummy_data(sector)
        return X, y, hist

    X_train = sector_df[features]
    y_train = sector_df[y_col]
    
    hist_data_df = pd.DataFrame(y_train).rename(columns={y_col: 'Target_Value'})
    
    return X_train, y_train, hist_data_df

def get_dummy_data(sector):
    """Fallback if data loading fails"""
    features = FEATURES_MAP[sector]
    dates = pd.date_range(start="2018-01-01", end="2024-12-31", freq='MS')
    n = len(dates)
    data = pd.DataFrame(index=dates)
    X = pd.DataFrame(index=data.index[1:], columns=features)
    base_time = np.arange(n-1); seasonal = np.sin(base_time * 2 * np.pi / 12) * 50
    X[features[0]] = 100 + base_time * 0.1 + seasonal + np.random.randn(n-1)*5
    X[features[1]] = 120 - base_time * 0.05 + seasonal + np.random.randn(n-1)*5
    X[features[2]] = 140 + (np.random.randn(n-1)*1.5).cumsum()
    X[features[3]] = 80 + seasonal * 0.5 + np.random.randn(n-1)*2
    y = 50 + np.random.randn(n-1)*2
    y += (X[features[0]] - 100) * 2.5; y -= (X[features[1]] - 120) * 2.0
    y -= (X[features[2]] - 140) * 1.5; y -= (X[features[3]] - 80) * 0.5
    data['Target_Value'] = np.nan; data.iloc[1:, data.columns.get_loc('Target_Value')] = y
    return X, y, data

def build_future_features(base_scenario, sector, features, n_future):
    """
    Creates the X_future DataFrame based on a scenario dictionary.
    """
    future_dates = pd.date_range(start="2025-01-01", periods=n_future, freq='MS')
    X_future = pd.DataFrame(index=future_dates, columns=features)

    rate_mod, oil_mod, wage_mod, mod_1, mod_2 = [base_scenario[k] for k in SCENARIO_LEVELS.keys()]
    
    fx_rate_change = (wage_mod * 0.8) - (rate_mod * 2.0)
    isk_eur_rate_future = 145 * (1 + fx_rate_change / 100)
    marine_fuel_future = 100 * (1 + (oil_mod * 1.5) / 100)
    jet_fuel_future = 120 * (1 + (oil_mod * 1.2) / 100)
    eu_sentiment_future = 100 * (1 + (wage_mod / 100 * 0.5) - (oil_mod / 100 * 0.2))
    global_prod_future = 100 * (1 + (wage_mod / 100 * 0.3) - (oil_mod / 100 * 0.3))
    electricity_cost_future = 50 * (1 - rate_mod / 100)
    
    future_driver_values = {
        "ISK_EUR_Rate_lag1": np.full(n_future, isk_eur_rate_future), # Use lagged name
        "Inflation_Annual_Pct_lag1": np.full(n_future, 5.0 * (1 + wage_mod/100)), # Simple inflation sim
        "Marine_Fuel_Price": np.full(n_future, marine_fuel_future),
        "Jet_Fuel_Price": np.full(n_future, jet_fuel_future),
        "EU_Economic_Sentiment": np.full(n_future, eu_sentiment_future),
        "Global_Industrial_Production": np.full(n_future, global_prod_future),
        "Electricity_Cost": np.full(n_future, electricity_cost_future),
        "seasonal_driver": np.sin(np.arange(n_future) * 2 * np.pi / 12) * 50
    }
    
    sector_feature_map = {
        "🐟 Cod Exports": {"mod1": "Cod_Quota_kT", "mod2": "Norwegian_Quota_kT"},
        "✈️ Tourism": {"mod1": "Airline_Seat_Capacity", "mod2": "EU_Economic_Sentiment"},
        "🏭 Aluminum": {"mod1": "LME_Aluminum_Price", "mod2": "Carbon_Credit_Price_EUR"},
    }
    
    for feature in features:
        if feature == sector_feature_map[sector]['mod1']:
            X_future[feature] = 110 * (1 + mod_1 / 100) + future_driver_values["seasonal_driver"]
        elif feature == sector_feature_map[sector]['mod2']:
            if feature == "Carbon_Credit_Price_EUR":
                 X_future[feature] = 80 * (1 + mod_2 / 100)
            else:
                 X_future[feature] = 110 * (1 + mod_2 / 100)
        elif feature in future_driver_values:
            X_future[feature] = future_driver_values[feature]
        else:
            X_future[feature] = 100
            
    return X_future

# --- 3. Main Execution ---
def main():
    print("Starting pre-computation of all scenarios...")
    
    master_df = load_all_data()
    
    scenario_keys = SCENARIO_LEVELS.keys()
    scenario_combinations = list(itertools.product(*SCENARIO_LEVELS.values()))
    
    all_forecasts = []
    baseline_shap_data = {}

    for sector in SECTORS:
        X_train, y_train, hist_data = get_training_data(sector, master_df)
        
        if y_train.empty:
            print(f"Skipping {sector} due to missing data.")
            continue
        
        model = RandomForestRegressor(n_estimators=30, random_state=42)
        model.fit(X_train, y_train)
        
        explainer = shap.TreeExplainer(model)
        baseline_shap_data[sector] = {
            'global_shap': explainer(X_train),
            'X_train': X_train,
            'base_value': explainer.expected_value
        }
        
        print(f"Pre-computing {len(scenario_combinations)} scenarios for {sector}...")
        for combo in scenario_combinations:
            scenario = dict(zip(scenario_keys, combo))
            
            X_future = build_future_features(scenario, sector, FEATURES_MAP[sector], n_future=24)
            predictions = model.predict(X_future)
            
            forecast_df = pd.DataFrame({
                'date': X_future.index,
                'forecast_value': predictions,
                'sector': sector
            })
            
            for key, value in scenario.items():
                forecast_df[key] = value
                
            all_forecasts.append(forecast_df)

        # Add historical data
        hist_df = pd.DataFrame({
            'date': hist_data.index,
            'historical_value': hist_data['Target_Value'],
            'sector': sector
        })
        all_forecasts.append(hist_df)

    final_df = pd.concat(all_forecasts, ignore_index=True)
    final_df.to_parquet("app/precomputed_forecasts.parquet", index=False)
    
    print("Pre-rendering baseline SHAP plots...")
    shap_plots = {}
    
    for sector in SECTORS:
        if sector not in baseline_shap_data:
            continue
            
        theme_text_color = '#E9ECEF'
        
        plt.rcdefaults() 
        shap.summary_plot(
            baseline_shap_data[sector]['global_shap'], 
            baseline_shap_data[sector]['X_train'], 
            plot_type="bar", show=False
        )
        fig_global = plt.gcf()
        fig_global.axes[0].set_title("Global Feature Importance (Baseline)")

        plt.rcParams['text.color'] = theme_text_color
        plt.rcParams['axes.labelcolor'] = theme_text_color
        plt.rcParams['xtick.color'] = theme_text_color
        plt.rcParams['ytick.color'] = theme_text_color
        plt.rcParams['figure.facecolor'] = 'none'
        plt.rcParams['axes.facecolor'] = 'none'
        plt.rcParams['axes.edgecolor'] = theme_text_color
        plt.rcParams['savefig.transparent'] = True

        baseline_scenario = dict(zip(scenario_keys, [0.0]*len(scenario_keys)))
        X_baseline = build_future_features(baseline_scenario, sector, FEATURES_MAP[sector], 24)
        shap_values_baseline = explainer(X_baseline)
        
        shap.waterfall_plot(
            shap.Explanation(
                base_values=baseline_shap_data[sector]['base_value'],
                values=shap_values_baseline.values[0],
                feature_names=FEATURES_MAP[sector],
                data=X_baseline.iloc[0]
            ),
            show=False
        )
        fig_local = plt.gcf()
        
        shap_plots[sector] = {
            'global_plot_src': convert_plt_to_base64(fig_global),
            'local_plot_src': convert_plt_to_base64_transparent(fig_local)
        }
    
    with open('app/baseline_shap_plots.json', 'w') as f:
        json.dump(shap_plots, f)
        
    print("--- Pre-computation Complete! ---")
    print(f"Saved {len(final_df)} rows to 'app/precomputed_forecasts.parquet'")
    print("Saved baseline SHAP plots to 'app/baseline_shap_plots.json'")

if __name__ == '__main__':
    main()