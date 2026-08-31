FROM python:3.11-slim

WORKDIR /app

# CPU 전용 torch (CUDA 이미지 대비 수 GB 절약)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY run.py .

EXPOSE 8899
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8899"]
