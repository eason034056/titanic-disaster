FROM python:3.13

# Set working directory
WORKDIR /app

# Copy requirements.txt to container
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy code to container
COPY . .

# Run Python script
CMD ["python", "titanic_analysis.py"]