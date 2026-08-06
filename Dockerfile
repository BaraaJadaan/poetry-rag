FROM python:3.13-slim

# Install system dependencies required for building C++ extensions (llama.cpp and tantivy)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /app

# Install uv globally
RUN pip install uv

# Copy dependency files first for caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv.
# We set CMAKE_ARGS to ensure llama.cpp builds optimally for the underlying CPU (e.g. ARM NEON).
ENV CMAKE_ARGS="-DGGML_NATIVE=ON"
RUN uv sync --no-dev

# Copy the rest of the application
COPY . /app

# Ensure we have the environment variables
ENV PYTHONUNBUFFERED=1

# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# Run the entrypoint script to handle downloading/embedding and starting the server
CMD ["./entrypoint.sh"]
