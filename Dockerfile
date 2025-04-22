FROM python:3.10-slim

# Set working directory  
WORKDIR /app  

# Copy requirements first (better caching)  
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt pytest  

# Copy the rest of the app  
COPY . .  

# Run the app  
CMD ["python", "app.py"]  
