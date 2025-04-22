
# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files to container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir nbconvert ipykernel

# Automatically execute the notebook when container runs
CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "Milk_Quality_Prediction.ipynb"]
