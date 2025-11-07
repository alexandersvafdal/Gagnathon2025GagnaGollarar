# --- Stage 1: Base Image ---
# Use the stable Python 3.12 image (was 3.13, but 3.12 is safer)
FROM python:3.13-slim

# --- Stage 2: Environment Setup ---
# Set the working directory inside the container
WORKDIR /app

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# --- Stage 3: Install Dependencies ---
# Copy requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 4: Copy Application Code ---
# Copy all your app code (app.py, etc.)
COPY . .

# --- Stage 5: Expose Port ---
# Expose the port Streamlit will run on
EXPOSE 8501

# --- Stage 6: Run Command ---
# The command to run your Streamlit app
# This includes all the flags for it to work in Docker
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]