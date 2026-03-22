import uuid, time, traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file, Response
from vtracer_engine import vectorize

BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC_DIR))

ALLOWED        = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}
MAX_FILE_BYTES = 20 * 1024 * 1024  # 20MB

@app.errorhandler(Exception)
def eany(e):
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

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

    t0 = time.time()
    svg = vectorize(
        raw,
        posterize_bits    = gi("posterize_bits",    7),
        unsharp_radius    = gf("unsharp_radius",    0.5),
        unsharp_percent   = gi("unsharp_percent",   90),
        unsharp_threshold = gi("unsharp_threshold", 4),
        colormode        = "color",
        hierarchical     = "stacked",
        mode             = "spline",
        filter_speckle   = gi("filter_speckle",   4),
        color_precision  = gi("color_precision",  8),
        layer_difference = gi("layer_difference", 2),
        corner_threshold = gi("corner_threshold", 60),
        length_threshold = gf("length_threshold", 4.0),
        max_iterations   = gi("max_iterations",   1),
        splice_threshold = gi("splice_threshold", 45),
        path_precision   = gi("path_precision",   1),
    )
    elapsed = round(time.time() - t0, 2)
    paths   = svg.count("<path")

    # Save to disk for PDF export endpoint only
    job_id   = uuid.uuid4().hex[:12]
    out_path = OUTPUT_DIR / f"{job_id}.svg"
    out_path.write_text(svg, encoding="utf-8")

    # Keep only last 20 outputs on disk
    svgs = sorted(OUTPUT_DIR.glob("*.svg"), key=lambda p: p.stat().st_mtime)
    for old in svgs[:-20]:
        old.unlink(missing_ok=True)

    # Return SVG inline — eliminates the second /api/preview fetch
    return jsonify({
        "job_id":   job_id,
        "elapsed":  elapsed,
        "paths":    paths,
        "svg":      svg,
        "download": f"/api/download/{job_id}",
    })

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
