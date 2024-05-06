# Use the official Python base image for Python 3.11
FROM python:3.11.8

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install additional dependencies
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the Python dependencies file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Run the application
CMD ["make", "run"]
