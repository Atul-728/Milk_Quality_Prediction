
# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files to container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir nbconvert ipykernel

# Run the Jupyter notebook on container start
CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "Milk_Quality_Prediction.ipynb"]
