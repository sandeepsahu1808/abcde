FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5081

CMD ["uvicorn", "src.anomaly.api:app", "--host", "0.0.0.0", "--port", "5081"]
