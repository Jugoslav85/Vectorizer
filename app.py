import uuid, time, traceback, hashlib, json, os, hmac as _hmac, secrets
from pathlib import Path
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify, send_from_directory, send_file, Response, make_response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from vtracer_engine import vectorize

BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "outputs"
STATIC_DIR  = BASE_DIR / "static"
SAMPLES_DIR = STATIC_DIR / "images" / "samples"
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC_DIR))
app.secret_key = os.environ.get("SECRET_KEY", "scaylr-dev-key-change-in-prod")
limiter = Limiter(get_remote_address, app=app, default_limits=[], storage_uri="memory://")


ALLOWED        = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}
MAX_FILE_BYTES = 20 * 1024 * 1024
CACHE_TTL      = 3600
_cache: dict   = {}

# Limit concurrent conversions to 1 per worker to prevent CPU starvation
import threading
_conversion_lock = threading.Semaphore(1)

_KEY_SECRET            = os.environ.get("LICENSE_SECRET",        "scaylr-key-secret-change-in-prod")
_STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
_RESEND_API_KEY        = os.environ.get("RESEND_API_KEY",        "")
_FROM_EMAIL            = os.environ.get("FROM_EMAIL",            "keys@scaylr.io")
_APP_URL               = os.environ.get("APP_URL",               "https://scaylr.io")
_DATABASE_URL          = os.environ.get("DATABASE_URL",          "")

# ── Database — works with Postgres (Railway) or SQLite (local dev) ─────────────

_USE_POSTGRES = bool(_DATABASE_URL)

if _USE_POSTGRES:
    import psycopg2
    import psycopg2.extras
    from psycopg2 import pool as _pg_pool
    print(f"[db] using Postgres", flush=True)
else:
    import sqlite3
    _DB_PATH = str(BASE_DIR / "scaylr.db")
    print(f"[db] using SQLite at {_DB_PATH}", flush=True)

# ── Connection pool (Postgres only) ───────────────────────────────────────────
_pg_connection_pool = None

def _get_pool():
    global _pg_connection_pool
    if _USE_POSTGRES and _pg_connection_pool is None:
        _pg_connection_pool = _pg_pool.ThreadedConnectionPool(
            minconn=1, maxconn=8, dsn=_DATABASE_URL
        )
        print("[db] connection pool initialised (max=8)", flush=True)
    return _pg_connection_pool


class _PooledConn:
    """Context manager that borrows/returns a Postgres pooled connection."""
    def __init__(self):
        self._pool = _get_pool()
        self._conn = None
    def __enter__(self):
        self._conn = self._pool.getconn()
        self._conn.autocommit = False
        return self._conn
    def __exit__(self, exc_type, *_):
        if exc_type:
            try: self._conn.rollback()
            except Exception: pass
        self._pool.putconn(self._conn)

def _db():
    """Return an open DB connection context manager."""
    if _USE_POSTGRES:
        return _PooledConn()
    else:
        import contextlib
        @contextlib.contextmanager
        def _sqlite_ctx():
            conn = sqlite3.connect(_DB_PATH)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            try:
                yield conn
            finally:
                conn.close()
        return _sqlite_ctx()

def _cursor(conn):
    """Return a dict-returning cursor for either backend."""
    if _USE_POSTGRES:
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    return conn.cursor()


def _q(sql: str) -> str:
    """Convert %s placeholders to ? for SQLite."""
    if _USE_POSTGRES:
        return sql
    return sql.replace("%s", "?")


def _row_to_dict(row) -> dict:
    """Normalise a DB row to a plain dict regardless of backend."""
    if row is None:
        return None
    if _USE_POSTGRES:
        return dict(row)
    return dict(row)


def _fetchone(cursor):
    row = cursor.fetchone()
    return _row_to_dict(row)


def _db_init():
    """Create tables — idempotent, safe to call on every startup."""
    with _db() as conn:
        cur = _cursor(conn)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pro_keys (
                key                    TEXT PRIMARY KEY,
                email                  TEXT NOT NULL,
                stripe_customer_id     TEXT,
                stripe_subscription_id TEXT UNIQUE,
                status                 TEXT NOT NULL DEFAULT 'active',
                created_at             TEXT NOT NULL,
                expires_at             TEXT NOT NULL,
                renewed_at             TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_subscription ON pro_keys(stripe_subscription_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_email ON pro_keys(email)")
        # Stripe event idempotency table — prevents duplicate key/email on webhook retry
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stripe_events (
                event_id   TEXT PRIMARY KEY,
                processed_at TEXT NOT NULL
            )
        """)
        conn.commit()
    print("[db] tables ready", flush=True)

_db_init()

# ── Key helpers ───────────────────────────────────────────────────────────────

def _generate_key() -> str:
    uid = secrets.token_hex(2).upper()
    sig = _hmac.new(_KEY_SECRET.encode(), uid.encode(), hashlib.sha256).hexdigest()[:12].upper()
    return f"SCAYLR-{uid}-{sig[:4]}-{sig[4:8]}-{sig[8:12]}"

def _hmac_valid(key: str) -> bool:
    try:
        parts = key.upper().strip().split("-")
        if len(parts) != 5 or parts[0] != "SCAYLR":
            return False
        uid = parts[1]
        sig_provided = parts[2] + parts[3] + parts[4]
        sig_expected = _hmac.new(
            _KEY_SECRET.encode(), uid.encode(), hashlib.sha256
        ).hexdigest()[:12].upper()
        return _hmac.compare_digest(sig_provided, sig_expected)
    except Exception:
        return False

def _validate_key(key: str) -> bool:
    if not _hmac_valid(key):
        return False
    try:
        with _db() as conn:
            cur = _cursor(conn)
            cur.execute(
                _q("SELECT status, expires_at FROM pro_keys WHERE key = %s"),
                (key.upper().strip(),)
            )
            row = _fetchone(cur)
        if not row:
            return False
        if row["status"] != "active":
            return False
        expires = datetime.fromisoformat(row["expires_at"])
        # Ensure timezone-aware comparison — treat naive datetimes as UTC
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return expires > datetime.now(timezone.utc)
    except Exception:
        return False

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _expires_iso(days: int = 37) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()

# ── Email via Resend ──────────────────────────────────────────────────────────

def _send_key_email(to_email: str, key: str, is_renewal: bool = False) -> bool:
    if not _RESEND_API_KEY:
        print(f"[email] RESEND_API_KEY not set — skipping email to {to_email}", flush=True)
        return False

    subject = "Your Scaylr Pro is renewed" if is_renewal else "Your Scaylr Pro key"

    if is_renewal:
        body_html = f"""<div style="font-family:sans-serif;max-width:520px;margin:0 auto;padding:32px 24px;color:#111">
  <h2 style="font-size:24px;font-weight:800;margin:0 0 8px">Subscription renewed</h2>
  <p style="color:#555;margin:0 0 28px">Your Scaylr Pro subscription has renewed. Your existing key keeps working — no action needed.</p>
  <div style="background:#f5f4ff;border:1.5px solid #d4d0fa;border-radius:12px;padding:20px 24px;margin-bottom:28px">
    <p style="font-size:12px;color:#888;margin:0 0 8px;text-transform:uppercase;letter-spacing:1px">Your license key</p>
    <p style="font-family:monospace;font-size:20px;font-weight:700;color:#5b4fd4;margin:0;letter-spacing:1px">{key}</p>
  </div>
  <p style="color:#555;margin:0 0 24px">This key has been extended for another month.</p>
  <a href="{_APP_URL}/app" style="display:inline-block;background:#5b4fd4;color:#fff;text-decoration:none;padding:13px 28px;border-radius:10px;font-weight:700;font-size:15px">Open Scaylr</a>
  <p style="color:#aaa;font-size:12px;margin:28px 0 0">Questions? Reply to this email.</p>
</div>"""
    else:
        body_html = f"""<div style="font-family:sans-serif;max-width:520px;margin:0 auto;padding:32px 24px;color:#111">
  <h2 style="font-size:24px;font-weight:800;margin:0 0 8px">Welcome to Scaylr Pro</h2>
  <p style="color:#555;margin:0 0 28px">Thanks for subscribing. Here's your license key.</p>
  <div style="background:#f5f4ff;border:1.5px solid #d4d0fa;border-radius:12px;padding:20px 24px;margin-bottom:28px">
    <p style="font-size:12px;color:#888;margin:0 0 8px;text-transform:uppercase;letter-spacing:1px">Your license key</p>
    <p style="font-family:monospace;font-size:20px;font-weight:700;color:#5b4fd4;margin:0;letter-spacing:1px">{key}</p>
  </div>
  <p style="color:#555;margin:0 0 8px"><strong>How to activate:</strong></p>
  <ol style="color:#555;margin:0 0 24px;padding-left:20px;line-height:1.8">
    <li>Go to <a href="{_APP_URL}/app" style="color:#5b4fd4">{_APP_URL}/app</a></li>
    <li>Click PDF or Fine-tune settings</li>
    <li>Click "Already have a key?" and paste your key</li>
  </ol>
  <a href="{_APP_URL}/app" style="display:inline-block;background:#5b4fd4;color:#fff;text-decoration:none;padding:13px 28px;border-radius:10px;font-weight:700;font-size:15px">Open Scaylr</a>
  <p style="color:#aaa;font-size:12px;margin:28px 0 0">Your key renews automatically each month. Questions? Reply to this email.</p>
</div>"""

    try:
        import requests as _req
        r = _req.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {_RESEND_API_KEY}", "Content-Type": "application/json"},
            json={"from": f"Scaylr <{_FROM_EMAIL}>", "to": [to_email],
                  "subject": subject, "html": body_html},
            timeout=10,
        )
        ok = r.status_code in (200, 201)
        print(f"[email] {'sent' if ok else 'failed'} to {to_email} ({r.status_code})", flush=True)
        return ok
    except Exception as e:
        print(f"[email] exception: {e}", flush=True)
        return False

# ── Stripe webhook ────────────────────────────────────────────────────────────

@app.route("/stripe-webhook", methods=["POST"])
def stripe_webhook():
    if request.content_length and request.content_length > 512 * 1024:
        return jsonify({"error": "Payload too large"}), 413
    payload    = request.get_data(limit=512 * 1024)
    sig_header = request.headers.get("Stripe-Signature", "")

    if _STRIPE_WEBHOOK_SECRET:
        try:
            import stripe as _stripe
            event = _stripe.Webhook.construct_event(payload, sig_header, _STRIPE_WEBHOOK_SECRET)
        except Exception as e:
            print(f"[webhook] sig check failed: {e}", flush=True)
            return jsonify({"error": "Invalid signature"}), 400
    else:
        try:
            event = json.loads(payload)
        except Exception:
            return jsonify({"error": "Bad JSON"}), 400

    event_type = event.get("type", "")
    event_id   = event.get("id", "")
    obj        = event.get("data", {}).get("object", {})
    print(f"[webhook] {event_type} ({event_id[:12]})", flush=True)

    # ── Idempotency check — skip already-processed events ─────────────────────
    if event_id:
        try:
            with _db() as conn:
                cur = _cursor(conn)
                cur.execute(_q("SELECT event_id FROM stripe_events WHERE event_id = %s"), (event_id,))
                if _fetchone(cur):
                    print(f"[webhook] duplicate event {event_id[:12]} — skipping", flush=True)
                    return jsonify({"ok": True})
                cur.execute(
                    _q("INSERT INTO stripe_events (event_id, processed_at) VALUES (%s, %s)"),
                    (event_id, _now_iso())
                )
                conn.commit()
        except Exception as e:
            print(f"[webhook] idempotency check error: {e}", flush=True)
            # Don't block processing if idempotency table fails

    if event_type == "invoice.paid":
        sub_id      = obj.get("subscription")
        customer_id = obj.get("customer")
        email       = (obj.get("customer_email") or
                       obj.get("customer_details", {}).get("email") or "")
        if not sub_id or not email:
            return jsonify({"ok": True})

        with _db() as conn:
            cur = _cursor(conn)
            cur.execute(
                _q("SELECT key FROM pro_keys WHERE stripe_subscription_id=%s"), (sub_id,)
            )
            existing = _fetchone(cur)

            if existing:
                # Renewal — extend expiry, same key unchanged
                key = existing["key"]
                cur.execute(
                    _q("UPDATE pro_keys SET status='active', expires_at=%s, renewed_at=%s WHERE key=%s"),
                    (_expires_iso(), _now_iso(), key)
                )
                conn.commit()
                print(f"[webhook] renewed {email}", flush=True)
                _send_key_email(email, key, is_renewal=True)
            else:
                # New subscription
                key = _generate_key()
                cur.execute(
                    _q("INSERT INTO pro_keys (key,email,stripe_customer_id,stripe_subscription_id,status,created_at,expires_at) VALUES (%s,%s,%s,%s,'active',%s,%s)"),
                    (key, email.lower(), customer_id, sub_id, _now_iso(), _expires_iso())
                )
                conn.commit()
                print(f"[webhook] new key for {email}", flush=True)
                _send_key_email(email, key, is_renewal=False)

    elif event_type == "customer.subscription.deleted":
        sub_id = obj.get("id")
        if sub_id:
            with _db() as conn:
                cur = _cursor(conn)
                cur.execute(
                    _q("UPDATE pro_keys SET status='revoked' WHERE stripe_subscription_id=%s"), (sub_id,)
                )
                conn.commit()
            print(f"[webhook] cancelled {sub_id}", flush=True)

    elif event_type in ("invoice.payment_failed", "invoice.payment_action_required"):
        sub_id = obj.get("subscription")
        if sub_id:
            with _db() as conn:
                cur = _cursor(conn)
                cur.execute(
                    _q("UPDATE pro_keys SET status='revoked' WHERE stripe_subscription_id=%s"), (sub_id,)
                )
                conn.commit()
            print(f"[webhook] payment failed {sub_id}", flush=True)

    return jsonify({"ok": True})

# ── Validate key ──────────────────────────────────────────────────────────────

@app.route("/api/resend-key", methods=["POST"])
def api_resend_key():
    data  = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    if not email or "@" not in email:
        return jsonify({"ok": False, "error": "Please enter a valid email address"}), 400
    try:
        with _db() as conn:
            cur = _cursor(conn)
            cur.execute(
                _q("SELECT key, status, expires_at FROM pro_keys WHERE email = %s ORDER BY expires_at DESC LIMIT 1"),
                (email,)
            )
            row = _fetchone(cur)
    except Exception as e:
        return jsonify({"ok": False, "error": "Database error"}), 500

    if not row:
        # Don't reveal whether email exists — just say sent
        return jsonify({"ok": True})

    from datetime import datetime, timezone
    try:
        expires = datetime.fromisoformat(row["expires_at"])
        if row["status"] != "active" or expires < datetime.now(timezone.utc):
            return jsonify({"ok": True})  # silently ignore revoked/expired
    except Exception:
        pass

    _send_key_email(email, row["key"], is_renewal=False)
    return jsonify({"ok": True})

@app.route("/api/validate-key", methods=["POST"])
def api_validate_key():
    data = request.get_json(silent=True) or {}
    key  = str(data.get("key", "")).strip().upper()
    if not key:
        return jsonify({"valid": False, "error": "No key provided"}), 400
    valid = _validate_key(key)
    if valid:
        try:
            with _db() as conn:
                cur = _cursor(conn)
                cur.execute(_q("SELECT expires_at FROM pro_keys WHERE key=%s"), (key,))
                row = _fetchone(cur)
            expires = row["expires_at"] if row else None
        except Exception:
            expires = None
        return jsonify({"valid": True, "plan": "pro", "expires_at": expires})
    return jsonify({"valid": False, "plan": "free"})

# ── Cache helpers ─────────────────────────────────────────────────────────────

def _get_session_id(req):
    return req.cookies.get("vsid") or uuid.uuid4().hex

def _cache_key_fn(image_bytes, settings):
    h = hashlib.md5(image_bytes).hexdigest()
    s = json.dumps(settings, sort_keys=True)
    return hashlib.md5(f"{h}:{s}".encode()).hexdigest()

def _cache_get(session_id, key):
    entry = _cache.get(session_id, {}).get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry
    return None

def _cache_set(session_id, key, value):
    if session_id not in _cache:
        _cache[session_id] = {}
    value["ts"] = time.time()
    _cache[session_id][key] = value
    now = time.time()
    for sid in list(_cache.keys()):
        _cache[sid] = {k: v for k, v in _cache[sid].items() if now - v["ts"] < CACHE_TTL}
        if not _cache[sid]:
            del _cache[sid]

@app.after_request
def add_cors(resp):
    origin = request.headers.get("Origin", "")
    allowed = os.environ.get("APP_URL", "https://scaylr.io")
    if origin in (allowed, allowed.replace("https://", "https://www.")):
        resp.headers["Access-Control-Allow-Origin"] = origin
    return resp

@app.errorhandler(Exception)
def eany(e):
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return "ok", 200

@app.route("/")
def landing():
    return send_from_directory(STATIC_DIR, "landing.html")

@app.route("/sitemap.xml")
def sitemap():
    return send_from_directory(BASE_DIR, "sitemap.xml"), 200, {"Content-Type": "application/xml"}

@app.route("/robots.txt")
def robots():
    return send_from_directory(BASE_DIR, "robots.txt"), 200, {"Content-Type": "text/plain"}

@app.route("/terms")
def terms():
    return send_from_directory(STATIC_DIR, "terms.html")

@app.route("/privacy")
def privacy():
    return send_from_directory(STATIC_DIR, "privacy.html")

@app.route("/favicon.svg")
def favicon():
    return send_from_directory(str(STATIC_DIR) + "/images", "favicon.svg"), 200, {"Content-Type": "image/svg+xml"}

@app.route("/app")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/api/samples")
def api_samples():
    meta_path = SAMPLES_DIR / "samples.json"
    if meta_path.exists():
        samples = json.loads(meta_path.read_text())
    else:
        exts  = {".png", ".jpg", ".jpeg", ".webp"}
        files = sorted(f for f in SAMPLES_DIR.iterdir() if f.suffix.lower() in exts)
        samples = [{"file": f.name, "name": f.stem.replace("-"," ").replace("_"," ").title(),
                    "preset": "illustration"} for f in files]
    for s in samples:
        s["url"] = f"/static/images/samples/{s['file']}"
    return jsonify(samples)

@app.route("/api/sample-image/<filename>")
def api_sample_image(filename):
    safe = Path(filename).name
    p = SAMPLES_DIR / safe
    if not p.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(p)

@app.route("/api/vectorize", methods=["POST"])
@limiter.limit("20 per hour;5 per minute")
def api_vectorize():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if Path(f.filename).suffix.lower() not in ALLOWED:
        return jsonify({"error": "Unsupported file type"}), 400
    raw = f.read()
    if len(raw) > MAX_FILE_BYTES:
        return jsonify({"error": "File too large (max 20MB)"}), 400
    try:
        from PIL import Image
        import io
        img_check = Image.open(io.BytesIO(raw))
        img_check.load()  # fully decode without destroying stream
        img_check.close()
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    def gi(k, d):
        try: return int(request.form.get(k, d))
        except: return d
    def gf(k, d):
        try: return float(request.form.get(k, d))
        except: return d

    settings = {
        "median_size":          gi("median_size",           3),
        "morph_close_size":     gi("morph_close_size",      3),
        "color_precision":      gi("color_precision",       6),
        "layer_difference":     gi("layer_difference",      4),
        "filter_speckle":       gi("filter_speckle",        6),
        "engine_mode":          request.form.get("engine_mode", "auto"),
        "posterize_bits":       gi("posterize_bits",        6),
        "simplify_epsilon":     gf("simplify_epsilon",      0.1),
        "corner_threshold":     gi("corner_threshold",      48),
        "splice_threshold":     gi("splice_threshold",      70),
        "max_colors":           gi("max_colors",            32),
        "guided_filter_radius": gi("guided_filter_radius",  4),
        "color_dedup_thresh":   gi("color_dedup_thresh",    12),
        "svg_dedup_thresh":     gi("svg_dedup_thresh",      10),
        # Post-processing options
        "gap_fill":             request.form.get("gap_fill", "1") == "1",
        "gap_fill_width":       gf("gap_fill_width",        1.5),
        "stroke_edges":         request.form.get("stroke_edges", "0") == "1",
        "stroke_edges_width":   gf("stroke_edges_width",    1.5),
        "stroke_edges_color":   request.form.get("stroke_edges_color", "") or None,
    }

    session_id = _get_session_id(request)
    ck         = _cache_key_fn(raw, settings)
    cached     = _cache_get(session_id, ck)

    if cached:
        resp = make_response(jsonify({
            "job_id":      cached["job_id"],
            "elapsed":     cached["elapsed"],
            "paths":       cached["paths"],
            "svg":         cached["svg"],
            "download":    f"/api/download/{cached['job_id']}",
            "cached":      True,
            "engine_mode": cached.get("engine_mode", "color"),
        }))
        resp.set_cookie("vsid", session_id, max_age=86400, samesite="Lax", httponly=True)
        return resp

    # Try to acquire the conversion slot — reject immediately if busy
    acquired = _conversion_lock.acquire(blocking=False)
    if not acquired:
        return jsonify({"error": "Server is busy processing another image. Please try again in a moment."}), 429

    t0 = time.time()
    try:
        svg = vectorize(
            raw,
            median_size          = settings["median_size"],
            morph_close_size     = settings["morph_close_size"],
            posterize_bits       = settings["posterize_bits"],
            engine_mode          = settings["engine_mode"],
            simplify             = True,
            simplify_epsilon     = settings["simplify_epsilon"],
            filter_speckle       = settings["filter_speckle"],
            color_precision      = settings["color_precision"],
            layer_difference     = settings["layer_difference"],
            corner_threshold     = settings["corner_threshold"],
            splice_threshold     = settings["splice_threshold"],
            max_colors           = settings["max_colors"],
            guided_filter_radius = settings["guided_filter_radius"],
            color_dedup_thresh   = settings["color_dedup_thresh"],
            svg_dedup_thresh     = settings["svg_dedup_thresh"],
            gap_fill             = settings["gap_fill"],
            gap_fill_width       = settings["gap_fill_width"],
            stroke_edges         = settings["stroke_edges"],
            stroke_edges_width   = settings["stroke_edges_width"],
            stroke_edges_color   = settings["stroke_edges_color"],
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {e}"}), 500
    finally:
        _conversion_lock.release()

    elapsed  = round(time.time() - t0, 2)
    paths    = svg.count("<path")
    job_id   = uuid.uuid4().hex[:12]
    out_path = OUTPUT_DIR / f"{job_id}.svg"
    out_path.write_text(svg, encoding="utf-8")

    # Prune old SVG files atomically to avoid race between workers
    try:
        svgs = sorted(OUTPUT_DIR.glob("*.svg"), key=lambda p: p.stat().st_mtime)
        for old in svgs[:-20]:
            old.unlink(missing_ok=True)
    except Exception:
        pass  # Non-fatal — pruning failure doesn't affect the response

    _cache_set(session_id, ck, {
        "job_id": job_id, "elapsed": elapsed, "paths": paths,
        "svg": svg, "engine_mode": settings["engine_mode"]
    })

    resp = make_response(jsonify({
        "job_id":      job_id,
        "elapsed":     elapsed,
        "paths":       paths,
        "svg":         svg,
        "download":    f"/api/download/{job_id}",
        "cached":      False,
        "engine_mode": settings["engine_mode"],
    }))
    resp.set_cookie("vsid", session_id, max_age=86400, samesite="Lax", httponly=True)
    return resp

@app.route("/api/download/<job_id>")
def api_download(job_id):
    if not job_id.isalnum():
        return jsonify({"error": "Bad ID"}), 400
    p = OUTPUT_DIR / f"{job_id}.svg"
    if not p.exists():
        return jsonify({"error": "Not found"}), 404
    resp = send_file(p, mimetype="image/svg+xml", as_attachment=True,
                     download_name=f"vector_{job_id}.svg")
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp

@app.route("/api/download-pdf/<job_id>")
def api_download_pdf(job_id):
    if not job_id.isalnum():
        return jsonify({"error": "Bad ID"}), 400
    key = request.headers.get("X-License-Key", "")
    if not _validate_key(key):
        return jsonify({"error": "Pro plan required for PDF export", "upgrade": True}), 403
    p = OUTPUT_DIR / f"{job_id}.svg"
    if not p.exists():
        return jsonify({"error": "Not found"}), 404
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
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
