FROM python:3.12
LABEL authors="EsotericMat"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt \
                --index-url https://download.pytorch.org/whl/cpu \
                --extra-index-url https://pypi.org/simple ## This one will force the installation of torch in a CPU version
EXPOSE 8080

CMD ["gunicorn", "inference.app:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080"]