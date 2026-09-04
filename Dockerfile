# Base image
FROM python:3.11

# Working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Document the application port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "serving.app.main:app", "--host", "0.0.0.0", "--port", "8000"]