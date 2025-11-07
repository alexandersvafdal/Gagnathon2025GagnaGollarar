# 🏆 HVDS Gagnathon 2025 - Team GagnaGöllarar 🏆

This repository contains our team's solution for the HVDS AI Gagnathon 2025: an interactive, explainable AI (XAI) dashboard for simulating and forecasting key sectors of the Icelandic economy.

---

### 🛠️ Tech Architecture

* **Model:** `RandomForestRegressor` (from `scikit-learn`) used as a robust and fast model for the interactive simulation.
* **Core Libraries:** `Pandas`, `Scikit-learn`, `Shap`, `Plotly`, `Matplotlib`
* **Frontend:** `Streamlit`
* **Language:** Python 🐍 3.13
* **Deployment:** `Docker`

---

### 🏃 How to Run

You can run the project either natively on your machine or inside a Docker container.

#### 1. Native Installation (Recommended for quick tests)

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Project:**
    ```bash
    streamlit run app.py
    ```

#### 2. Run with Docker (Recommended for stable deployment)

This is the best way to ensure the environment is identical for all users.

1.  **Build the Docker Image:**
    Make sure you have Docker Desktop installed and running. From the project's root directory (the one with the `Dockerfile`), run:
    ```bash
    docker build -t iceland-simulator .
    ```

2.  **Run the Docker Container:**
    ```bash
    docker run -p 8501:8501 iceland-simulator
    ```

3.  **Access the App:**
    Open your browser and go to **http://localhost:8501**

4.  **Build and Run the Docker Container with Docker Compose:**
    Make sure you have Docker Desktop installed and running. From the project's root directory, run:
    ```bash
    docker-compose up --build
    ```

5.  **Access the App:**
    Open your browser and go to **http://localhost:8050**

6.  **Stop the App:**
    To stop the app, press `Ctrl+C` in the terminal where Docker Compose is running.

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
* **Look (1-5):** We created a clean, colorful, and understandable interface using Streamlit and Plotly.
* **Usability (1-5):** The tool is highly interactive, using simple sliders to control complex macro-economic and sectoral levers.
* **Clarity (1-5):** The project's purpose is clear. The "Explainability (SHAP)" tab provides direct insight into *why* the model is making its predictions.
* **Ease of putting in regular production (1-5):** Our code is organized and includes a `requirements.txt` and a fully functional `Dockerfile` for one-command deployment.
* **User relevance (1-5):** The tool provides a high-value simulation dashboard for economists, academics, and policymakers (like our target judges from Statistics Iceland) to test scenarios.
---

### 👥 Team Members
* Alexander Svarfdal Guðmundsson
* Birgir Snær Ingason
* Elvar Þór Sævarsson
* Sigurður Baldursson
