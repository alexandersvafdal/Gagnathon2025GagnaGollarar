import pandas as pd
import numpy as np
import itertools
import warnings

# --- AutoGluon TimeSeries ---
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
PREDICTION_LENGTH = 24       # months ahead
FREQ = "MS"                  # Month Start
SECTORS = ["🐟 Cod Exports", "✈️ Tourism", "🏭 Aluminum"]

# Map: sector -> known covariate column names (must exist in master_df)
FEATURES_MAP = {
    "🐟 Cod Exports": [
        "Cod_Quota_kT",
        "Norwegian_Quota_kT",
        "ISK_EUR_Rate_lag1",
        "Inflation_Annual_Pct_lag1",
    ],
    "✈️ Tourism": [
        "Airline_Seat_Capacity",
        "ISK_EUR_Rate_lag1",
        "EU_Economic_Sentiment",
        "Inflation_Annual_Pct_lag1",
    ],
    "🏭 Aluminum": [
        "LME_Aluminum_Price",
        "Carbon_Credit_Price_EUR",
        "Global_Industrial_Production",
        "ISK_EUR_Rate_lag1",
    ],
}

# Scenario knobs (percent deltas)
SCENARIO_LEVELS = {
    "rate_mod": [-1.0, 0.0, 1.0],       # central bank rate shock proxy
    "oil_mod": [-20.0, 0.0, 20.0],      # oil price proxy
    "wage_mod": [-5.0, 0.0, 5.0],       # wage index proxy
    "sector_mod_1": [-20.0, 0.0, 20.0], # sector driver 1
    "sector_mod_2": [-20.0, 0.0, 20.0], # sector driver 2
}

# =========================
# DATA LOADING  (same spirit as before)
# =========================
def load_fish_exports():
    try:
        df = pd.read_parquet("data/Exports-by-branches-of-processing-and-countries-2015-2025.parquet")
        df_fob = df[df["Unit"] == "Fob value"].copy()
        df_fob["date"] = pd.to_datetime(df_fob["Month"], format="%YM%m")
        df_fob["DATA"] = pd.to_numeric(df_fob["DATA"], errors="coerce").fillna(0)
        df_agg = df_fob.groupby("date")["DATA"].sum().reset_index()
        df_agg = df_agg.set_index("date").resample(FREQ).sum()
        df_agg = df_agg.rename(columns={"DATA": "Fish_Export_Value_FOB"})
        return df_agg[["Fish_Export_Value_FOB"]]
    except Exception as e:
        print(f"Fish export file not found or error: {e}. Using dummy data.")
        dates = pd.date_range("2018-01-01", "2024-12-31", freq=FREQ)
        return pd.DataFrame(index=dates, data={"Fish_Export_Value_FOB": np.random.rand(len(dates))*1000 + 50000})

def load_tourism_exports():
    try:
        df = pd.read_csv("data/External trade in services, monthly 2018-2025.csv", sep=";", header=2)
        df = df.rename(columns={df.columns[0]: "Service"})
        df_travel = df[df["Service"].str.contains("Travel", case=False, na=False)].copy()
        df_long = df_travel.melt(id_vars=["Service"], var_name="month_str", value_name="Tourism_Export_Value")
        df_long["date"] = pd.to_datetime(df_long["month_str"], format="%YM%m", errors="coerce")
        df_long = df_long.dropna(subset=["date"])
        df_long["Tourism_Export_Value"] = pd.to_numeric(
            df_long["Tourism_Export_Value"].astype(str).str.replace(".", "", regex=False).str.replace(",", "."),
            errors="coerce"
        )
        df_agg = df_long.groupby("date")[["Tourism_Export_Value"]].sum().resample(FREQ).sum()
        return df_agg
    except Exception as e:
        print(f"Tourism export file not found or error: {e}. Using dummy data.")
        dates = pd.date_range("2018-01-01", "2024-12-31", freq=FREQ)
        return pd.DataFrame(index=dates, data={"Tourism_Export_Value": np.random.rand(len(dates))*1000 + 20000})

def load_macro_features():
    try:
        df_inf = pd.read_csv("data/Inflation-Consumer price index.csv")
        df_inf["date"] = pd.to_datetime(df_inf["Month"].str.strip('"'), format="%YM%m")
        df_inf = df_inf.set_index("date")
        df_inf = df_inf[["Consumer price index Annual change, %"]].rename(
            columns={"Consumer price index Annual change, %": "Inflation_Annual_Pct"}
        )

        df_fx = pd.read_csv(
            "data/Exchange-rates_2015-2025.csv", sep=";", decimal=",",
            usecols=["Dagsetning", "Evra EUR miðgengi", "Bandaríkjadalur USD miðgengi"]
        )
        df_fx["date"] = pd.to_datetime(df_fx["Dagsetning"], format="%d.%m.%Y")
        df_fx = df_fx.set_index("date")
        df_fx_monthly = df_fx[["Evra EUR miðgengi", "Bandaríkjadalur USD miðgengi"]].resample(FREQ).mean()
        df_fx_monthly = df_fx_monthly.rename(
            columns={"Evra EUR miðgengi": "ISK_EUR_Rate", "Bandaríkjadalur USD miðgengi": "ISK_USD_Rate"}
        )
        df_macro = df_inf.join(df_fx_monthly, how="outer")
        return df_macro
    except Exception as e:
        print(f"Macro data files not found or error: {e}. Using dummy data.")
        dates = pd.date_range("2018-01-01", "2024-12-31", freq=FREQ)
        return pd.DataFrame(index=dates, data={
            "Inflation_Annual_Pct": np.random.rand(len(dates))*5,
            "ISK_EUR_Rate": np.random.rand(len(dates))*20 + 130,
        })

def load_all_data():
    df_fish = load_fish_exports()
    df_tourism = load_tourism_exports()
    df_macro = load_macro_features()

    # Dummy Aluminum series (placeholder)
    dates = pd.date_range("2018-01-01", "2024-12-31", freq=FREQ)
    df_aluminum = pd.DataFrame(index=dates, data={"Aluminum_Export_Value": np.random.rand(len(dates))*1000 + 30000})

    df_master = df_macro.join(df_fish, how="outer").join(df_tourism, how="outer").join(df_aluminum, how="outer")

    # Lags
    df_master["ISK_EUR_Rate_lag1"] = df_master["ISK_EUR_Rate"].shift(1)
    df_master["Inflation_Annual_Pct_lag1"] = df_master["Inflation_Annual_Pct"].shift(1)

    # Sectoral drivers (toy)
    df_master["Cod_Quota_kT"] = 150 + np.sin(df_master.index.month * 2 * np.pi / 12) * 5
    df_master["Norwegian_Quota_kT"] = 200 + np.sin(df_master.index.month * 2 * np.pi / 12) * 10
    df_master["Airline_Seat_Capacity"] = 500000 + np.sin(df_master.index.month * 2 * np.pi / 12) * 100000
    df_master["EU_Economic_Sentiment"] = 100 + np.random.randn(len(df_master)) * 2
    df_master["LME_Aluminum_Price"] = 2000 + np.random.randn(len(df_master)) * 100
    df_master["Carbon_Credit_Price_EUR"] = 80 + np.random.randn(len(df_master)) * 5
    df_master["Global_Industrial_Production"] = 100 + np.random.randn(len(df_master)) * 2

    # Drop rows before lags exist
    df_master = df_master.dropna(subset=["ISK_EUR_Rate_lag1", "Inflation_Annual_Pct_lag1"])
    return df_master

# =========================
# HELPERS
# =========================
def get_sector_columns(sector: str):
    if sector == "🐟 Cod Exports":
        y_col = "Fish_Export_Value_FOB"
    elif sector == "✈️ Tourism":
        y_col = "Tourism_Export_Value"
    elif sector == "🏭 Aluminum":
        y_col = "Aluminum_Export_Value"
    else:
        raise ValueError(f"Unknown sector: {sector}")
    x_cols = FEATURES_MAP[sector]
    return y_col, x_cols

def make_tsdf_single_series(target_series: pd.Series, sector: str) -> TimeSeriesDataFrame:
    """
    Build a TimeSeriesDataFrame for a single series using a MultiIndex (item_id, timestamp)
    and a single column 'target' as required by newer AG TS.
    """
    df = target_series.rename("target").to_frame()
    df["item_id"] = sector
    df["timestamp"] = df.index
    df = df.set_index(["item_id", "timestamp"]).sort_index()
    return TimeSeriesDataFrame(df)

def make_covariates_tsdf(cov_df: pd.DataFrame, sector: str) -> TimeSeriesDataFrame:
    """
    Known covariates TimeSeriesDataFrame. Columns = known covariate names. Index must be (item_id, timestamp).
    """
    df = cov_df.copy()
    df["item_id"] = sector
    df["timestamp"] = df.index
    df = df.set_index(["item_id", "timestamp"]).sort_index()
    return TimeSeriesDataFrame(df)

def build_future_features(base_scenario: dict, sector: str, features: list[str], n_future: int) -> pd.DataFrame:
    future_dates = pd.date_range(start="2025-01-01", periods=n_future, freq=FREQ)
    X_future = pd.DataFrame(index=future_dates, columns=features, dtype=float)

    rate_mod, oil_mod, wage_mod, mod_1, mod_2 = [base_scenario[k] for k in SCENARIO_LEVELS.keys()]

    fx_rate_change = (wage_mod * 0.8) - (rate_mod * 2.0)
    isk_eur_rate_future = 145 * (1 + fx_rate_change / 100)
    marine_fuel_future = 100 * (1 + (oil_mod * 1.5) / 100)
    jet_fuel_future = 120 * (1 + (oil_mod * 1.2) / 100)
    eu_sentiment_future = 100 * (1 + (wage_mod / 100 * 0.5) - (oil_mod / 100 * 0.2))
    global_prod_future = 100 * (1 + (wage_mod / 100 * 0.3) - (oil_mod / 100 * 0.3))
    electricity_cost_future = 50 * (1 - rate_mod / 100)

    future_driver_values = {
        "ISK_EUR_Rate_lag1": np.full(n_future, isk_eur_rate_future),
        "Inflation_Annual_Pct_lag1": np.full(n_future, 5.0 * (1 + wage_mod / 100)),
        "Marine_Fuel_Price": np.full(n_future, marine_fuel_future),
        "Jet_Fuel_Price": np.full(n_future, jet_fuel_future),
        "EU_Economic_Sentiment": np.full(n_future, eu_sentiment_future),
        "Global_Industrial_Production": np.full(n_future, global_prod_future),
        "Electricity_Cost": np.full(n_future, electricity_cost_future),
        "seasonal_driver": np.sin(np.arange(n_future) * 2 * np.pi / 12) * 50,
    }

    sector_feature_map = {
        "🐟 Cod Exports": {"mod1": "Cod_Quota_kT", "mod2": "Norwegian_Quota_kT"},
        "✈️ Tourism": {"mod1": "Airline_Seat_Capacity", "mod2": "EU_Economic_Sentiment"},
        "🏭 Aluminum": {"mod1": "LME_Aluminum_Price", "mod2": "Carbon_Credit_Price_EUR"},
    }

    for feature in features:
        if feature == sector_feature_map[sector]["mod1"]:
            X_future[feature] = 110 * (1 + base_scenario["sector_mod_1"] / 100) + future_driver_values["seasonal_driver"]
        elif feature == sector_feature_map[sector]["mod2"]:
            if feature == "Carbon_Credit_Price_EUR":
                X_future[feature] = 80 * (1 + base_scenario["sector_mod_2"] / 100)
            else:
                X_future[feature] = 110 * (1 + base_scenario["sector_mod_2"] / 100)
        elif feature in future_driver_values:
            X_future[feature] = future_driver_values[feature]
        else:
            X_future[feature] = 100.0
    return X_future

# =========================
# MAIN
# =========================
def main():
    print("Starting pre-computation with AutoGluon TimeSeries...")
    master_df = load_all_data()

    scenario_keys = list(SCENARIO_LEVELS.keys())
    scenario_combinations = list(itertools.product(*SCENARIO_LEVELS.values()))

    all_outputs = []

    for sector in SECTORS:
        y_col, x_cols = get_sector_columns(sector)

        # Drop rows where target or any covariate is missing
        df_sector = master_df.dropna(subset=[y_col] + x_cols)
        if df_sector.empty:
            print(f"Skipping {sector}: no valid rows after dropping NA.")
            continue

        # === Build training target TSDF ===
        y_train = df_sector[y_col]
        ts_train = make_tsdf_single_series(y_train, sector)

        # === Known covariates for history ===
        X_hist = df_sector[x_cols]
        kc_hist = make_covariates_tsdf(X_hist, sector)
        
        # === Merge covariates into the training TSDF ===
        ts_train_full = ts_train.join(kc_hist, how="left")

        # === Create & fit predictor ===
        predictor = TimeSeriesPredictor(
            prediction_length=PREDICTION_LENGTH,
            freq=FREQ,
            known_covariates_names=x_cols,
            eval_metric="MAPE",
        )
        predictor.fit(train_data=ts_train_full)

        print(f"Pre-computing {len(scenario_combinations)} scenarios for {sector}...")

        # === Scenarios ===
        for combo in scenario_combinations:
            scenario = dict(zip(scenario_keys, combo))

            # Future covariates
            X_future = build_future_features(scenario, sector, x_cols, n_future=PREDICTION_LENGTH)

            # Combine all known covariates: FULL history + future
            kc_all_df = pd.concat([X_hist, X_future], axis=0)
            kc_all = make_covariates_tsdf(kc_all_df, sector)

            # Forecast
            preds = predictor.predict(ts_train_full, known_covariates=kc_all)

            # Convert to pandas; select this sector
            preds_pd = preds.to_pandas().xs(sector, level="item_id")

            # Point forecast column
            if "mean" in preds_pd.columns:
                yhat = preds_pd["mean"]
            elif "0.5" in preds_pd.columns:
                yhat = preds_pd["0.5"]
            else:
                yhat = preds_pd.iloc[:, 0]

            forecast_df = pd.DataFrame({
                "date": yhat.index,
                "forecast_value": yhat.values,
                "sector": sector,
            })
            for k, v in scenario.items():
                forecast_df[k] = v
            all_outputs.append(forecast_df)

        # Append historical series for plotting/back-end joins
        hist_df = pd.DataFrame({
            "date": y_train.index,
            "historical_value": y_train.values,
            "sector": sector,
        })
        all_outputs.append(hist_df)

    final_df = pd.concat(all_outputs, ignore_index=True)
    final_df.to_parquet("precomputed_forecasts.parquet", index=False)

    print("--- Pre-computation Complete! ---")
    print(f"Saved {len(final_df)} rows to 'precomputed_forecasts.parquet'")

if __name__ == "__main__":
    main()