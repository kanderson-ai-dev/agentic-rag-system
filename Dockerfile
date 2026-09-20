FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies (no dev/test tooling).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code.
COPY app/ app/
COPY frontend/ frontend/
COPY evaluation/ evaluation/

# Run as a non-root user with a writable data directory.
RUN useradd --create-home appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

# Healthcheck via the Python stdlib (no curl in slim images).
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
