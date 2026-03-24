"""
Scaylr license key generator.

Usage:
    python generate_keys.py [count]

Keys are HMAC-SHA256 signed — no database required.
Format: SCAYLR-{UID4}-{SIG4}-{SIG4}-{SIG4}

IMPORTANT: Set LICENSE_SECRET env var in Railway before generating
production keys. Keys generated with different secrets are incompatible.

    export LICENSE_SECRET="your-long-random-secret-here"
    python generate_keys.py 10
"""

import hmac, hashlib, secrets, os, sys

SECRET = os.environ.get("LICENSE_SECRET", "scaylr-key-secret-change-in-prod")

if SECRET == "scaylr-key-secret-change-in-prod":
    print("⚠️  WARNING: Using default secret. Set LICENSE_SECRET env var for production keys.\n")

def generate_key() -> str:
    uid = secrets.token_hex(2).upper()          # 4 hex chars
    sig = hmac.new(SECRET.encode(), uid.encode(), hashlib.sha256).hexdigest()[:12].upper()
    return f"SCAYLR-{uid}-{sig[:4]}-{sig[4:8]}-{sig[8:12]}"

def validate_key(key: str) -> bool:
    try:
        parts = key.upper().strip().split("-")
        if len(parts) != 5 or parts[0] != "SCAYLR":
            return False
        uid = parts[1]
        sig_provided = parts[2] + parts[3] + parts[4]
        sig_expected = hmac.new(SECRET.encode(), uid.encode(), hashlib.sha256).hexdigest()[:12].upper()
        return hmac.compare_digest(sig_provided, sig_expected)
    except Exception:
        return False

count = int(sys.argv[1]) if len(sys.argv) > 1 else 5

print(f"Generating {count} license key(s) with secret: {SECRET[:8]}...\n")
for _ in range(count):
    key = generate_key()
    print(f"  {key}")

print(f"\nAll keys valid: {all(validate_key(generate_key()) for _ in range(100))}")
