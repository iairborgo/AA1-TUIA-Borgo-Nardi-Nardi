FROM python:3.12-slim

WORKDIR /app

COPY Inferencia/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY Inferencia/ ./

COPY pipe/ ./pipe

ENTRYPOINT ["python", "inferencia.py"]
