# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.13.6-slim as builder

# Set environment variables to avoid python buffering output and creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies into a specific location (--user)
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ==========================================
# Stage 2: Runtime
# ==========================================
FROM python:3.13.6-slim as runtime

# Create a non-root user for security
RUN useradd -m appuser
USER appuser

WORKDIR /app

# Copy the installed python packages from the builder stage
# We ensure 'appuser' owns these files
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Update PATH so Python can find the installed packages
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app

# Copy the source code
COPY --chown=appuser:appuser src/ ./src/

# -----------------------------------------------------------
# DATASET STRATEGY: OPTION 1 (EMBEDDED)
# Copy the dataset directly. (Adjust 'data/' to your actual folder or filename)
# -----------------------------------------------------------
COPY --chown=appuser:appuser dataset.csv ./data/dataset.csv

# Expose the entry point
ENTRYPOINT ["python"]
CMD ["-m", "src.inference", "--help"]