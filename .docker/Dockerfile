# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install Poetry
RUN pip install poetry

# Disable virtual environments creation by Poetry
# This is recommended when building Docker images
# as it ensures the packages are installed globally in the image
RUN poetry config virtualenvs.create false

# Install the project dependencies
RUN poetry install --no-dev

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the application
CMD ["poetry", "run", "python", "-m", "dqn_reinforcement_learning"]
