# Base image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
# Install dependencies
# Document the application port
EXPOSE 8080

# Use the environment variable provided by Vertex AI, defaulting to 8080
CMD ["sh", "-c", "uvicorn serving.app.main:app --host 0.0.0.0 --port ${AIP_HTTP_PORT:-8080}"]