# --- Stage 1: Base Image ---
FROM python:3.12-slim

# --- Stage 2: Environment Setup ---
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DASH_DEBUG_MODE=False

WORKDIR /app

# --- Stage 3: Install Dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 4: Copy Application Code ---
COPY . .

# --- Stage 5: Expose Port ---
EXPOSE 8050

# --- Stage 6: Run Command ---
CMD ["gunicorn", "-b", "0.0.0.0:8050", "app.app:server"]