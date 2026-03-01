# =============================================================================
# FLAPS (Flight Lateness Australia Prediction System) - Docker Image
# =============================================================================
#
# Build:  docker build -t flaps-app .
# Run:    docker run -p 8501:8501 flaps-app
# Open:   http://localhost:8501
#
# =============================================================================

# --- Base Image ---
# Using Python 3.11 slim variant for smaller image size (~150MB vs ~900MB full)
FROM python:3.11-slim

# --- System Dependencies ---
# Install graphviz for keras.utils.plot_model (neural network architecture diagram)
# Clean up apt cache in same layer to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Working Directory ---
WORKDIR /app

# --- Python Dependencies ---
# Copy requirements first (Docker layer caching optimization)
# If requirements.txt hasn't changed, this layer is cached on rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application Code ---
# Copy everything else (code, models, data), except for what is in .dockerignore!
# This layer changes frequently, so it comes after dependencies
COPY . .

# --- Port Configuration ---
# Streamlit default port (documentation only, doesn't publish the port)
EXPOSE 8080

# --- Health Check ---
# Uses $PORT which Cloud Run sets to 8080 by default
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8080}/_stcore/health || exit 1

# --- Startup Command ---
# Run Streamlit with production settings:
#   --server.port=$PORT       : Use Cloud Run's PORT env variable (default 8080)
#   --server.address=0.0.0.0  : Accept connections from outside container
#   --server.headless=true    : Don't try to open browser
#   --browser.gatherUsageStats=false : Disable telemetry
CMD streamlit run app/Home.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
