FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy tests and install pytest
COPY tests /app/tests/
RUN pip install pytest  # Explicitly install pytest

# Copy the rest of the application
COPY . .

# Command to run the application
CMD ["python", "app.py"]
