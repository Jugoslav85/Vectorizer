import uuid, time, traceback, hashlib, json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file, Response, make_response
from vtracer_engine import vectorize

BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "outputs"
STATIC_DIR  = BASE_DIR / "static"
SAMPLES_DIR = STATIC_DIR / "images" / "samples"
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC_DIR))
app.secret_key = "vect-secret-change-in-prod"

ALLOWED        = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}
MAX_FILE_BYTES = 20 * 1024 * 1024  # 20MB
CACHE_TTL      = 3600              # 1 hour in seconds

# ── In-memory cache ──────────────────────────────────────────────────────────
# Structure: { session_id: { cache_key: { svg, paths, elapsed, job_id, ts } } }
_cache: dict = {}

def _get_session_id(req) -> str:
    """Get or create a session ID from cookie."""
    return req.cookies.get("vsid") or uuid.uuid4().hex

def _cache_key(image_bytes: bytes, settings: dict) -> str:
    h = hashlib.md5(image_bytes).hexdigest()
    s = json.dumps(settings, sort_keys=True)
    return hashlib.md5(f"{h}:{s}".encode()).hexdigest()

def _cache_get(session_id: str, key: str):
    session = _cache.get(session_id, {})
    entry   = session.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry
    return None

def _cache_set(session_id: str, key: str, value: dict):
    if session_id not in _cache:
        _cache[session_id] = {}
    value["ts"] = time.time()
    _cache[session_id][key] = value
    _prune_cache()

def _prune_cache():
    """Remove expired entries to keep memory clean."""
    now = time.time()
    for sid in list(_cache.keys()):
        _cache[sid] = {
            k: v for k, v in _cache[sid].items()
            if now - v["ts"] < CACHE_TTL
        }
        if not _cache[sid]:
            del _cache[sid]

# ── Error handler ─────────────────────────────────────────────────────────────
@app.errorhandler(Exception)
def eany(e):
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500

# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route("/")
def landing():
    return send_from_directory(STATIC_DIR, "landing.html")

@app.route("/remove-bg")
def remove_bg_page():
    return send_from_directory(STATIC_DIR, "remove_bg.html")

@app.route("/api/remove-bg", methods=["POST"])
def api_remove_bg():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    raw = f.read()
    if len(raw) > MAX_FILE_BYTES:
        return jsonify({"error": "File too large (max 20MB)"}), 400
    try:
        from rembg import remove, new_session
        from io import BytesIO
        import threading
        # Lazy-load session
        if not hasattr(api_remove_bg, '_session'):
            api_remove_bg._session = new_session('birefnet-general')
        result = remove(raw, session=api_remove_bg._session)
        return Response(result, mimetype="image/png")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/app")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

# ── Samples API ───────────────────────────────────────────────────────────────
@app.route("/api/samples")
def api_samples():
    """Return sample list from samples.json, or auto-discover if no json."""
    meta_path = SAMPLES_DIR / "samples.json"
    if meta_path.exists():
        samples = json.loads(meta_path.read_text())
    else:
        # Auto-discover image files if no samples.json
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        files = sorted(f for f in SAMPLES_DIR.iterdir() if f.suffix.lower() in exts)
        samples = [
            {"file": f.name, "name": f.stem.replace("-", " ").replace("_", " ").title(), "preset": "illustration"}
            for f in files
        ]
    # Attach URLs
    for s in samples:
        s["url"] = f"/static/images/samples/{s['file']}"
    return jsonify(samples)

@app.route("/api/sample-image/<filename>")
def api_sample_image(filename):
    """Serve a sample image file directly."""
    safe = Path(filename).name  # strip any path traversal
    p = SAMPLES_DIR / safe
    if not p.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(p)

# ── Vectorize ─────────────────────────────────────────────────────────────────
@app.route("/api/vectorize", methods=["POST"])
def api_vectorize():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if Path(f.filename).suffix.lower() not in ALLOWED:
        return jsonify({"error": "Unsupported file type"}), 400
    raw = f.read()
    if len(raw) > MAX_FILE_BYTES:
        return jsonify({"error": "File too large (max 20MB)"}), 400

    def gi(k, d):
        try: return int(request.form.get(k, d))
        except: return d
    def gf(k, d):
        try: return float(request.form.get(k, d))
        except: return d

    settings = {
        "blur_radius":     gf("blur_radius",     0.8),
        "color_precision": gi("color_precision",  8),
        "layer_difference":gi("layer_difference", 1),
        "filter_speckle":  gi("filter_speckle",   6),
        "remove_bg":       1 if request.form.get("remove_bg") == "1" else 0,
    }

    # ── Cache lookup ──
    session_id = _get_session_id(request)
    key        = _cache_key(raw, settings)
    cached     = _cache_get(session_id, key)

    if cached:
        print(f'[cache] HIT for session {session_id[:8]}', flush=True)
        resp = make_response(jsonify({
            "job_id":   cached["job_id"],
            "elapsed":  cached["elapsed"],
            "paths":    cached["paths"],
            "svg":      cached["svg"],
            "download": f"/api/download/{cached['job_id']}",
            "cached":   True,
        }))
        resp.set_cookie("vsid", session_id, max_age=86400, samesite="Lax")
        return resp

    # ── Process ──
    print(f'[cache] MISS for session {session_id[:8]}', flush=True)
    t0 = time.time()
    remove_bg = request.form.get("remove_bg") == "1"
    svg = vectorize(
        raw,
        posterize_bits    = 7,
        unsharp_radius    = 0.5,
        unsharp_percent   = 90,
        unsharp_threshold = 4,
        blur_radius       = settings["blur_radius"],
        remove_bg         = remove_bg,
        colormode         = "color",
        hierarchical      = "stacked",
        mode              = "spline",
        corner_threshold  = 1,
        length_threshold  = 3.5,
        max_iterations    = 1,
        splice_threshold  = 1,
        path_precision    = 1,
        filter_speckle    = settings["filter_speckle"],
        color_precision   = settings["color_precision"],
        layer_difference  = settings["layer_difference"],
    )
    elapsed = round(time.time() - t0, 2)
    paths   = svg.count("<path")

    # Save to disk for PDF export
    job_id   = uuid.uuid4().hex[:12]
    out_path = OUTPUT_DIR / f"{job_id}.svg"
    out_path.write_text(svg, encoding="utf-8")

    # Keep last 20 on disk
    svgs = sorted(OUTPUT_DIR.glob("*.svg"), key=lambda p: p.stat().st_mtime)
    for old in svgs[:-20]:
        old.unlink(missing_ok=True)

    # Store in cache
    _cache_set(session_id, key, {
        "job_id": job_id, "elapsed": elapsed, "paths": paths, "svg": svg
    })

    resp = make_response(jsonify({
        "job_id":   job_id,
        "elapsed":  elapsed,
        "paths":    paths,
        "svg":      svg,
        "download": f"/api/download/{job_id}",
        "cached":   False,
    }))
    resp.set_cookie("vsid", session_id, max_age=86400, samesite="Lax")
    return resp

# ── Download / PDF ────────────────────────────────────────────────────────────
@app.route("/api/download/<job_id>")
def api_download(job_id):
    if not job_id.isalnum(): return jsonify({"error": "Bad ID"}), 400
    p = OUTPUT_DIR / f"{job_id}.svg"
    if not p.exists(): return jsonify({"error": "Not found"}), 404
    return send_file(p, mimetype="image/svg+xml", as_attachment=True,
                     download_name=f"vector_{job_id}.svg")

@app.route("/api/download-pdf/<job_id>")
def api_download_pdf(job_id):
    if not job_id.isalnum(): return jsonify({"error": "Bad ID"}), 400
    p = OUTPUT_DIR / f"{job_id}.svg"
    if not p.exists(): return jsonify({"error": "Not found"}), 404
    try:
        import cairosvg
        from io import BytesIO
        pdf_bytes = cairosvg.svg2pdf(url=str(p))
        return send_file(BytesIO(pdf_bytes), mimetype="application/pdf",
                         as_attachment=True, download_name=f"vector_{job_id}.pdf")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"PDF export failed: {e}"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"Vectorizer → http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)
