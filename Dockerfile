# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files to container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir nbconvert ipykernel flask

# Run the Flask app on container start
CMD ["python", "app.py"]