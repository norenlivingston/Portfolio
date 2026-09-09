FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Train on container start (keeps the image reproducible from source — no
# pretrained artifacts baked in) then serve. `mlruns.db`/`data/` persist to
# whatever volume is mounted at /app, if any.
CMD ["sh", "-c", "python run_pipeline.py && python 03_mlops/serve.py"]
