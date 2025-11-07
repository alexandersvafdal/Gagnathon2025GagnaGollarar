# --- Stage 1: Base Image ---
FROM python:3.12-slim

# --- Stage 2: Environment Setup ---
# Set environment variables for Python and Dash
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DASH_DEBUG_MODE False

# Set the working directory
WORKDIR /app

# --- Stage 3: Install Dependencies ---
# Copy requirements file first to leverage Docker's layer caching
# This assumes requirements.txt is in your project root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 4: Copy Application Code ---
# Copy the rest of your project's files into the container
# This copies app.py, precomputed_forecasts.parquet, baseline_shap_plots.json,
# and the 'app' folder (containing 'assets') all into /app
COPY . .

# --- Stage 5: Expose Port ---
EXPOSE 8050

# --- Stage 6: Run Command ---
# Use Gunicorn to run the app by pointing to the 'server' object in 'app.py'
# (I've removed '--reload' as it's for development, not production)
CMD ["gunicorn", "-b", "0.0.0:8050", "app:server"]