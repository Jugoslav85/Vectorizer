FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p outputs uploads

EXPOSE 8080

CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --graceful-timeout 300 app:app
