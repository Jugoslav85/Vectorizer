import hmac, hashlib

KEY    = "SCAYLR-C28A-6480-1661-083B"
SECRET = "scaylr-prod-2026-[a4ghopWrj8]"

parts = KEY.upper().strip().split("-")
print(f"Parts: {parts}")
print(f"Length: {len(parts)} (need 5)")
print(f"Prefix: {parts[0]} (need SCAYLR)")

uid          = parts[1]
sig_provided = parts[2] + parts[3] + parts[4]
sig_expected = hmac.new(SECRET.encode(), uid.encode(), hashlib.sha256).hexdigest()[:12].upper()

print(f"\nUID:          {uid}")
print(f"Sig provided: {sig_provided}")
print(f"Sig expected: {sig_expected}")
print(f"Match:        {hmac.compare_digest(sig_provided, sig_expected)}")
print(f"\nConclusion: Key is {'VALID' if hmac.compare_digest(sig_provided, sig_expected) else 'INVALID'} for this secret")
