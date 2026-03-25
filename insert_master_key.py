"""
Run this with: railway run python3 insert_master_key.py
It connects directly to your Railway Postgres using DATABASE_URL.
"""
import os, hmac, hashlib, secrets
from datetime import datetime, timezone

DATABASE_URL = os.environ.get("DATABASE_URL", "")
if not DATABASE_URL:
    raise SystemExit("ERROR: DATABASE_URL not set. Run with: railway run python3 insert_master_key.py")

# ── Your details ──────────────────────────────────────────────────
KEY   = "SCAYLR-C28A-6480-1661-083B"
EMAIL = "jugoslav34@gmail.com"
# ─────────────────────────────────────────────────────────────────

import psycopg2

now     = datetime.now(timezone.utc).isoformat()
expires = "2099-01-01T00:00:00+00:00"

conn = psycopg2.connect(DATABASE_URL)
cur  = conn.cursor()

cur.execute("""
    INSERT INTO pro_keys
        (key, email, stripe_customer_id, stripe_subscription_id,
         status, created_at, expires_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (key) DO UPDATE
        SET status='active', expires_at=%s
""", (KEY, EMAIL, "owner", "owner", "active", now, expires, expires))

conn.commit()
cur.close()
conn.close()

print(f"Done — master key inserted:")
print(f"  Key:     {KEY}")
print(f"  Email:   {EMAIL}")
print(f"  Expires: {expires}")
