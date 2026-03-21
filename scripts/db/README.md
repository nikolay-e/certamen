# Database Setup for Authentication

## Quick Start

```bash
# 1. Create PostgreSQL database
createdb certamen

# 2. Set environment variables
export CERTAMEN_DB_HOST=localhost
export CERTAMEN_DB_PORT=5432
export CERTAMEN_DB_NAME=certamen
export CERTAMEN_DB_USER=certamen
export CERTAMEN_DB_PASSWORD=your_password

# 3. Generate JWT secret
export CERTAMEN_JWT_SECRET=$(openssl rand -base64 32)

# 4. Run migration
python scripts/db/init_auth_db.py
```

## Docker Setup

```bash
docker run -d \
  --name certamen-postgres \
  -e POSTGRES_DB=certamen \
  -e POSTGRES_USER=certamen \
  -e POSTGRES_PASSWORD=your_password \
  -p 5432:5432 \
  postgres:16-alpine
```

## Tables Created

- `users` - User accounts (username, password hash, is_admin flag)
- `refresh_tokens` - Refresh token storage (SHA256 hashes, expiration, revocation)

## Development Mode

Skip authentication checks (for local development only):

```bash
export CERTAMEN_SKIP_AUTH=true
certamen gui
```
