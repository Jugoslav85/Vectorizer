import hmac, hashlib, psycopg2

DATABASE_URL = "postgresql://postgres:YfYvJLQBZTePTCkMyWJSGjLMFHjUesIX@crossover.proxy.rlwy.net:14598/railway"

# Check all keys in DB and test every likely secret variation
conn = psycopg2.connect(DATABASE_URL)
cur  = conn.cursor()
cur.execute("SELECT key, status, expires_at FROM pro_keys")
rows = cur.fetchall()
conn.close()

print(f"Keys in DB: {len(rows)}")
for row in rows:
    print(f"  {row[0]}  status={row[1]}  expires={row[2]}")

print()

secrets_to_try = [
    "scaylr-prod-2026-[a4ghopWrj8]",
    "scaylr-prod-2026-a4ghopWrj8",
    "scaylr-prod-2026-[a4ghopWrj8",
    "scaylr-prod-2026-a4ghopWrj8]",
    "scaylr-key-secret-change-in-prod",
    "scaylr-prod-2026-[a4ghopWrj8] ",
    " scaylr-prod-2026-[a4ghopWrj8]",
]

for key_row in rows:
    KEY = key_row[0]
    parts = KEY.upper().strip().split("-")
    uid          = parts[1]
    sig_provided = parts[2] + parts[3] + parts[4]
    print(f"Testing key: {KEY}  (uid={uid}, sig={sig_provided})")
    found = False
    for secret in secrets_to_try:
        sig = hmac.new(secret.encode(), uid.encode(), hashlib.sha256).hexdigest()[:12].upper()
        match = hmac.compare_digest(sig, sig_provided)
        if match:
            print(f"  ✓ MATCHES secret: '{secret}'")
            found = True
    if not found:
        print(f"  ✗ No secret matched — key was generated with an unknown secret")
    print()
