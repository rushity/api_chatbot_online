FROM python:3.10-slim

# Prevent Python buffering (important for logs)
ENV PYTHONUNBUFFERED=1

# App directory inside container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Cloud Run always uses port 8080
EXPOSE 8080

# Start Flask using Gunicorn
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "1"]
