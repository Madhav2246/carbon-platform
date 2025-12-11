# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY model_artifacts/carbon_model.pkl .
COPY model_artifacts/metrics.json .
COPY label_encoder.pkl .
COPY scaler.pkl .


# Copy application
COPY app.py .

# Set environment variable for Cloud Run
ENV PORT=8080

# Run with gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app