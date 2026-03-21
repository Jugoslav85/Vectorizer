FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p outputs uploads

EXPOSE 8080

CMD python -c "import os; from app import app; app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))"
