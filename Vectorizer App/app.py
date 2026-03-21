import uuid, time, traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file, Response
from vtracer_engine import vectorize

BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC_DIR))

ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}

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
        return jsonify({"error": "Bad file type"}), 400
    raw = f.read()

    def gi(k, d): 
        try: return int(request.form.get(k, d))
        except: return d
    def gf(k, d):
        try: return float(request.form.get(k, d))
        except: return d
    def gs(k, d):
        return str(request.form.get(k, d) or d)

    t0 = time.time()
    svg = vectorize(
        raw,
        colormode        = gs("colormode", "color"),
        hierarchical     = gs("hierarchical", "stacked"),
        mode             = gs("mode", "spline"),
        filter_speckle   = gi("filter_speckle", 4),
        color_precision  = gi("color_precision", 8),
        layer_difference = gi("layer_difference", 2),
        corner_threshold = gi("corner_threshold", 60),
        length_threshold = gf("length_threshold", 4.0),
        max_iterations   = gi("max_iterations", 10),
        splice_threshold = gi("splice_threshold", 45),
        path_precision   = gi("path_precision", 3),
    )
    elapsed = round(time.time() - t0, 2)

    job_id   = uuid.uuid4().hex[:12]
    out_path = OUTPUT_DIR / f"{job_id}.svg"
    out_path.write_text(svg, encoding="utf-8")

    return jsonify({
        "job_id":   job_id,
        "elapsed":  elapsed,
        "preview":  f"/api/preview/{job_id}",
        "download": f"/api/download/{job_id}",
    })

@app.route("/api/preview/<job_id>")
def api_preview(job_id):
    if not job_id.isalnum(): return jsonify({"error": "Bad ID"}), 400
    p = OUTPUT_DIR / f"{job_id}.svg"
    if not p.exists(): return jsonify({"error": "Not found"}), 404
    return Response(p.read_text(encoding="utf-8"), mimetype="image/svg+xml")

@app.route("/api/download/<job_id>")
def api_download(job_id):
    if not job_id.isalnum(): return jsonify({"error": "Bad ID"}), 400
    p = OUTPUT_DIR / f"{job_id}.svg"
    if not p.exists(): return jsonify({"error": "Not found"}), 404
    return send_file(p, mimetype="image/svg+xml", as_attachment=True,
                     download_name=f"vector_{job_id}.svg")

if __name__ == "__main__":
    print("Vectorizer → http://localhost:5000")
    app.run(debug=False, port=5000)
