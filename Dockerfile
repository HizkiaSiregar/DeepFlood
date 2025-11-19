# Use Python 3.10 slim image to keep size down
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create the uploads directory
RUN mkdir -p uploads

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Environment variable to ensure output is logged immediately
ENV PYTHONUNBUFFERED=1

# Run the application using Gunicorn (Production server) or Python
# Since your app.py uses app.run(), we will modify the command slightly
# We override the port to use the $PORT environment variable provided by Cloud Run
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app