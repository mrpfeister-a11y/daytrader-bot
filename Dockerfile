FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
# Trains on first start (needs ~2 min), then trades on schedule forever.
CMD ["python", "trade.py"]
