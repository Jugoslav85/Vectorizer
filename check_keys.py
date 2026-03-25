import psycopg2

DATABASE_URL = "postgresql://postgres:YfYvJLQBZTePTCkMyWJSGjLMFHjUesIX@crossover.proxy.rlwy.net:14598/railway"

conn = psycopg2.connect(DATABASE_URL)
cur  = conn.cursor()

cur.execute("SELECT key, email, status, expires_at, stripe_subscription_id FROM pro_keys ORDER BY expires_at DESC")
rows = cur.fetchall()

print(f"Total keys in database: {len(rows)}\n")
for row in rows:
    print(f"  Key:      {row[0]}")
    print(f"  Email:    {row[1]}")
    print(f"  Status:   {row[2]}")
    print(f"  Expires:  {row[3]}")
    print(f"  Sub ID:   {row[4]}")
    print()

cur.close()
conn.close()
