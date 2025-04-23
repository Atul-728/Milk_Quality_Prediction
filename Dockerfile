# Use official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy files to container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the notebook (convert and execute)
CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "Milk_Quality_Prediction.ipynb"]