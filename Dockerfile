# Use the official Python 3.13 'slim' image as our base
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file *first*
# This takes advantage of Docker's layer caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code into the container
COPY . .

# Tell Docker that the container listens on port 8501
EXPOSE 8501

# The command to run your Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]