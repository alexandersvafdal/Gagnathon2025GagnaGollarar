# HVDS Gagnathon 2025 - Team GagnaGöllarar - Second place

This repository contains our team's solution for the HVDS AI Gagnathon 2025: an interactive, explainable AI (XAI) dashboard for simulating and forecasting key sectors of the Icelandic economy.

---

### 🛠️ Tech Architecture

* **Model:** `RandomForestRegressor` (from `scikit-learn`) used as a robust and fast model for the interactive simulation.
* **Core Libraries:** `Pandas`, `Scikit-learn`, `Shap`, `Plotly`, `Matplotlib`
* **Frontend:** `Dash`
* **Language:** Python 🐍
* **Package management:** `uv` (uses `pyproject.toml` + `uv.lock` for reproducible installs)
* **Deployment:** `Docker` (refer to `Dockerfile` / `docker-compose.yml`)

---

### 🏃 How to Run

You can run the project either natively on your machine for development or inside a Docker container for a stable, reproducible deployment.

#### 1. Native Installation (Recommended for development)

This project uses `uv` for dependency resolution (see `pyproject.toml` and `uv.lock`).

1.  **Install the `uv` CLI (if you don't already have it):**

    Follow the [official `uv` installation instructions](https://docs.astral.sh/uv/getting-started/installation/) for your platform. 
    For example, `uv` can be installed via pip:

    ```bash
    pip install uv
    ```

2.  **Sync dependencies (creates a virtual environment and installs locked dependencies):**

    ```bash
    uv sync
    ```

    This will create a `.venv` in the project root and install the dependencies specified in `uv.lock`/`pyproject.toml`.

3.  **Activate the virtual environment:**

    ```bash
    source .venv/bin/activate
    ```

4.  **Run the Dash app:**

    We run the Dash app via Gunicorn (the same entrypoint used in the Docker image):

    ```bash
    gunicorn -b 0.0.0.0:8050 app.app:server
    ```

    Open `http://localhost:8050` in your browser to view the app.


#### 2. Run with Docker (Recommended for stable deployment)

This is the best way to ensure the environment is identical for all users.

1.  **Build the Docker Image:**
    From the project's root directory (the one with the `Dockerfile`), run:
    ```bash
    docker build -t iceland-simulator .
    ```

2.  **Run the Docker Container:**
    ```bash
    docker run -p 8050:8050 iceland-simulator
    ```

3.  **Access the App:**
    Open your browser and go to **http://localhost:8050**

4.  **Build and Run the Docker Container with Docker Compose:**
    From the project's root directory, run:
    ```bash
    docker-compose up --build
    ```

5.  **Access the App:**
    Open your browser and go to **http://localhost:8050**

5.  **Stop the App:**
    To stop the app, press `Ctrl+C` in the terminal where Docker or Docker Compose is running.

> **Browser Warning (macOS Users):**
>
> You may see a **blank white page** if you open the app in **Safari** while it's running in Docker.
>
> This is a known issue related to how Safari's WebKit engine handles WebSocket connections when they are proxied through Docker on `localhost`. The app is running correctly, but Safari is failing to connect.
>
> **✅ Solution:** Please use **Google Chrome** or **Firefox** to view the app when running it via Docker. It will work perfectly.

---

### ✅ Judging Criteria Addressed

Our solution was built to meet the datathon's official criteria:
* **Functionality (1-5):** Our model successfully trains and generates dynamic, real-time "what-if" forecasts.
* **Look (1-5):** We created a clean, colorful, and understandable interface using Dash and Plotly.
* **Usability (1-5):** The tool is highly interactive, using simple sliders to control complex macro-economic and sectoral levers.
* **Clarity (1-5):** The project's purpose is clear. The "Explainability (SHAP)" tab provides direct insight into *why* the model is making its predictions.
* **Ease of putting in regular production (1-5):** Our code is organized and includes `pyproject.toml`, `uv.lock` and a fully functional `Dockerfile` for one-command deployment.
* **User relevance (1-5):** The tool provides a high-value simulation dashboard for economists, academics, and policymakers (like our target judges from Statistics Iceland) to test scenarios.
---

### 📊 Data Format Specification

> **📄 For detailed data format documentation with visual diagrams, see [DATA_FORMAT.md](DATA_FORMAT.md)**

### Quick Reference

The application expects two Parquet files in the `app/` directory:

#### `historical_data.parquet`
Contains historical export data with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime64 | Date of the observation (monthly data) |
| `Country` | string | Export destination country |
| `Branches` | string | Industry sector/branch (e.g., "FISH TOTAL", "AL TOTAL") |
| `y_true` | float64 | Actual export value in millions ISK |
| `prediction_type` | string | Should be `null` or empty for historical data |

#### `all_predictions.parquet`
Contains model predictions with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime64 | Date of the prediction (monthly data) |
| `Country` | string | Export destination country |
| `Branches` | string | Industry sector/branch |
| `y_pred` | float64 | Predicted export value in millions ISK |
| `y_true` | float64 | Actual value (for validation period) or `null` (for future predictions) |
| `prediction_type` | string | Either "Validation" (past data with known actuals) or "Future" (forecast) |

**Notes:**
- Both files are merged on load, with the `value` column computed as `y_true.fillna(y_pred)`
- All monetary values are in **millions of Icelandic Króna (ISK)**
- The `timestamp` should be the first day of each month
- Missing values in `y_true` indicate future forecasts without known outcomes

**Example of generating the files:**
```python
import pandas as pd

# Historical data example
historical_df = pd.DataFrame({
    'timestamp': pd.date_range('2015-01-01', '2024-12-01', freq='MS'),
    'Country': 'Netherlands',
    'Branches': 'FISH TOTAL',
    'y_true': [...],  # actual values
    'prediction_type': None
})
historical_df.to_parquet('app/historical_data.parquet')

# Predictions example
predictions_df = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', '2026-12-01', freq='MS'),
    'Country': 'Netherlands',
    'Branches': 'FISH TOTAL',
    'y_pred': [...],  # predicted values
    'y_true': [...],  # actuals for validation period, NaN for future
    'prediction_type': ['Validation'] * 24 + ['Future'] * 24
})
predictions_df.to_parquet('app/all_predictions.parquet')
```

---

### �👥 Team Members
* Alexander Svarfdal Guðmundsson
* Birgir Snær Ingason
* Elvar Þór Sævarsson
* Sigurður Baldursson
