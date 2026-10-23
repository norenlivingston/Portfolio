FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Serving only. Training is a separate one-shot step (the `train` service in
# docker-compose.yml, or `python run_pipeline.py`), so starting the API never
# silently retrains. With no trained model present the API fails fast.
ENV PIPELINE_CONFIG=config.docker.yaml
CMD ["python", "03_mlops/serve.py"]
