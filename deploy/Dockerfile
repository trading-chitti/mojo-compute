# Mojo Compute Service - Production Dockerfile
# 2-space indentation as per project standards

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
  curl \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install Mojo SDK (when publicly available)
# For now, using Python implementations with TODO for Mojo migration
# RUN curl -fsSL https://get.modular.com | sh
# RUN modular install mojo

# Create app user
RUN groupadd -r trading-chitti && \
  useradd -r -g trading-chitti -s /bin/bash trading-chitti

# Set working directory
WORKDIR /app

# Copy requirements (if we create one)
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=trading-chitti:trading-chitti . .

# Create socket directory
RUN mkdir -p /var/run/mojo-compute && \
  chown trading-chitti:trading-chitti /var/run/mojo-compute

# Switch to app user
USER trading-chitti

# Expose socket path via volume
VOLUME ["/var/run/mojo-compute"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python3 /app/deploy/healthcheck.py || exit 1

# Start server
CMD ["python3", "server.py"]
