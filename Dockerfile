FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PORT=7860
ENV CHECKPOINT_PATH=artifacts/bs16/best_efficientnet_b0.pth
ENV MODEL_URL=https://huggingface.co/spaces/ngoctannguyen/ai-image-detector/resolve/main/artifacts/bs16/best_efficientnet_b0.pth

CMD ["bash", "-lc", "python scripts/bootstrap_model.py && uvicorn backend.api:app --host 0.0.0.0 --port ${PORT}"]
