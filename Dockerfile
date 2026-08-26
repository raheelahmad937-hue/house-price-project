# Dockerfile - the recipe for building our container

# Step 1: Start with a computer that already has Python installed
FROM python:3.11-slim

# Step 2: Make a folder inside the container called "app" and go into it
WORKDIR /app

# Step 3: Copy our shopping list in first, and install everything on it
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy ALL our project files into the container
COPY . .

# Step 5: Tell Docker "this container will listen on port 5000"
EXPOSE 5000

# Step 6: The command that runs when the container starts
CMD ["python", "app.py"]