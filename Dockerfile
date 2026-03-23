FROM python:3.11-slim

WORKDIR /app

# System deps for cairosvg, rembg, and Rust for vtracer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-xlib-2.0-0 \
    libgl1 \
    libglib2.0-0 \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download rembg model so it's baked into the image
# Avoids slow cold-start on first user request
RUN python3 -c "from rembg import new_session; new_session('birefnet-general')"

COPY . .

RUN mkdir -p outputs uploads

EXPOSE 8080

CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --graceful-timeout 300 app:app
