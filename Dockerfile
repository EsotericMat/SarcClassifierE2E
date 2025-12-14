FROM python:3.12
LABEL authors="EsotericMat"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["gunicorn", "inference.app:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080"]