FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTORCH_NO_CUDA_MEMORY_CACHING=1
ENV OMP_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]