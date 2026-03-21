# Vectorizer — Bitmap to SVG

A local vectorization app powered by **libpotrace**, **Flask**, and **Pillow**.  
Upload a raster image, tune the settings, and get a clean scalable SVG back.

## Requirements

- Python 3.10+
- libpotrace (C library)
- Flask, Pillow, NumPy

## Setup

### 1. Install libpotrace

**Ubuntu / Debian**
```bash
sudo apt install libpotrace-dev
```

**macOS**
```bash
brew install potrace
```

**Windows** — download the DLL from https://potrace.sourceforge.net and place it
alongside `app.py`, or install via MSYS2.

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Project structure

```
vectorizer/
├── app.py              # Flask server & REST API
├── potrace_engine.py   # Vectorization engine (ctypes → libpotrace)
├── requirements.txt
├── static/
│   └── index.html      # Full frontend (single file)
├── uploads/            # Temp incoming files
└── outputs/            # Generated SVG files
```

## API

**POST /api/vectorize**

| Field          | Type   | Default   | Description                         |
|----------------|--------|-----------|-------------------------------------|
| `file`         | file   | —         | Image to vectorize                  |
| `mode`         | string | `color`   | `color` or `bw`                     |
| `color`        | string | `#000000` | Foreground color (BW mode)          |
| `bg_color`     | string | ``        | Background color (empty=transparent)|
| `threshold`    | int    | `128`     | BW threshold (0-255)                |
| `turdsize`     | int    | `2`       | Speckle suppression                 |
| `alphamax`     | float  | `1.0`     | Corner smoothing (0-1.33)           |
| `opttolerance` | float  | `0.2`     | Curve optimisation tolerance        |
| `num_colors`   | int    | `8`       | Colors for color mode (2-32)        |

**Response**
```json
{
  "job_id":  "abc123",
  "svg":     "<svg>...</svg>",
  "elapsed": 0.42,
  "download": "/api/download/abc123"
}
```

**GET /api/download/{job_id}** — returns the SVG file.
