# Authentication for Certamen GUI

Certamen GUI now supports JWT-based authentication with PostgreSQL backend, similar to the lingua-quiz project's shared-database pattern.

## Features

- **JWT Access Tokens** - 15-minute expiration, HS256 algorithm
- **Refresh Tokens** - 30-day expiration, stored as SHA256 hashes in database
- **PostgreSQL Connection Pool** - Shared database connections for optimal performance
- **bcrypt Password Hashing** - Industry-standard password security
- **Protected Endpoints** - `/api/execute`, `/api/validate`, `/ws` require authentication
- **Public Endpoints** - `/health`, `/api/models`, `/api/auth/*` accessible without auth
- **Development Mode** - Skip authentication with `CERTAMEN_SKIP_AUTH=true`

## Quick Setup

### 1. Install Dependencies

```bash
pip install -e .
```

Dependencies added:

- `psycopg2-binary~=2.9` - PostgreSQL adapter
- `bcrypt~=4.0` - Password hashing
- `pyjwt~=2.8` - JWT token generation/verification
- `pydantic~=2.0` - Request/response validation

### 2. Setup Database

```bash
# Create PostgreSQL database
createdb certamen

# Set environment variables
export CERTAMEN_DB_HOST=localhost
export CERTAMEN_DB_PORT=5432
export CERTAMEN_DB_NAME=certamen
export CERTAMEN_DB_USER=certamen
export CERTAMEN_DB_PASSWORD=your_secure_password

# Generate JWT secret (REQUIRED for production)
export CERTAMEN_JWT_SECRET=$(openssl rand -base64 32)

# Run database migration
python scripts/db/init_auth_db.py
```

### 3. Run GUI Server

```bash
certamen gui
```

## API Endpoints

### Authentication

**Register**

```bash
POST /api/auth/register
Content-Type: application/json

{
  "username": "myuser",
  "password": "your_secure_password"  # pragma: allowlist secret
}

Response: 201 Created
{
  "token": "eyJ0eXAi...",
  "refresh_token": "xYz123...",
  "expires_in": "15m",
  "user": {
    "id": 1,
    "username": "myuser",
    "is_admin": false
  }
}
```

**Login**

```bash
POST /api/auth/login
Content-Type: application/json

{
  "username": "myuser",
  "password": "secure_password_123"  # pragma: allowlist secret
}

Response: 200 OK
{
  "token": "eyJ0eXAi...",
  "refresh_token": "xYz123...",
  "expires_in": "15m",
  "user": {
    "id": 1,
    "username": "myuser",
    "is_admin": false
  }
}
```

**Refresh Token**

```bash
POST /api/auth/refresh
Content-Type: application/json

{
  "refresh_token": "xYz123..."
}

Response: 200 OK
{
  "token": "eyJ0eXAi...",  // New access token
  "refresh_token": "aBc456...",  // New refresh token
  "expires_in": "15m",
  "user": {
    "id": 1,
    "username": "myuser",
    "is_admin": false
  }
}
```

**Delete Account**

```bash
DELETE /api/auth/delete-account
Authorization: Bearer eyJ0eXAi...

Response: 200 OK
{
  "message": "Account deleted successfully"
}
```

### Protected Endpoints

All protected endpoints require `Authorization: Bearer <token>` header:

```bash
# Execute workflow
POST /api/execute
Authorization: Bearer eyJ0eXAi...
Content-Type: application/json

{
  "nodes": [...],
  "edges": [...]
}

# Validate workflow
POST /api/validate
Authorization: Bearer eyJ0eXAi...
Content-Type: application/json

{
  "nodes": [...],
  "edges": [...]
}

# WebSocket connection
ws://localhost:8765/ws
# Send Authorization header during WebSocket handshake
```

## Environment Variables

```bash
# Database Configuration
CERTAMEN_DB_HOST=localhost
CERTAMEN_DB_PORT=5432
CERTAMEN_DB_NAME=certamen
CERTAMEN_DB_USER=certamen
CERTAMEN_DB_PASSWORD=your_password

# Connection Pool
CERTAMEN_DB_POOL_MIN_SIZE=2
CERTAMEN_DB_POOL_MAX_SIZE=10

# JWT Configuration
CERTAMEN_JWT_SECRET=your_jwt_secret_here  # REQUIRED
CERTAMEN_JWT_ACCESS_TOKEN_EXPIRES_MINUTES=15
CERTAMEN_JWT_REFRESH_TOKEN_EXPIRES_DAYS=30

# Development Options
SKIP_DB_INIT=false  # Skip database pool initialization
CERTAMEN_SKIP_AUTH=false  # Skip authentication checks
```

## Development Mode

For local development without database:

```bash
export CERTAMEN_SKIP_AUTH=true
certamen gui
```

This disables authentication checks and allows all requests.

**WARNING**: Never use `CERTAMEN_SKIP_AUTH=true` in production!

## Database Schema

### users table

- `id` - Serial primary key
- `username` - Unique username (3-50 chars)
- `password` - bcrypt hashed password
- `is_admin` - Admin flag
- `created_at` - Account creation timestamp
- `updated_at` - Last update timestamp (auto-updated)

### refresh_tokens table

- `id` - Serial primary key
- `user_id` - Foreign key to users.id (CASCADE delete)
- `token_hash` - SHA256 hash of refresh token
- `expires_at` - Token expiration timestamp
- `revoked_at` - Token revocation timestamp (NULL if active)
- `created_at` - Token creation timestamp

Indexes:

- `idx_refresh_tokens_token_hash` - Fast token lookup
- `idx_refresh_tokens_user_id` - Fast user tokens lookup
- `idx_refresh_tokens_expires_at` - Fast expired token cleanup

## Security Features

1. **Password Hashing** - bcrypt with automatic salt generation
2. **JWT Tokens** - HS256 algorithm, includes exp/iat/nbf/jti claims
3. **Refresh Token Rotation** - Old refresh token revoked on refresh
4. **Token Expiration** - Access tokens expire after 15 minutes
5. **Database Cleanup** - Expired/revoked tokens can be cleaned up periodically
6. **SQL Injection Prevention** - Parameterized queries throughout
7. **Connection Pooling** - Prevents connection exhaustion attacks

## Architecture

Based on lingua-quiz shared-database pattern:

```
┌─────────────┐
│   Client    │
└─────┬───────┘
      │ HTTP/WebSocket + JWT
      ▼
┌─────────────────────┐
│  aiohttp Server     │
│  + auth_middleware  │
└──────┬──────────────┘
       │
       ▼
┌──────────────────────┐
│  Auth Module         │
│  - JWT verification  │
│  - Token generation  │
│  - Password hashing  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Database Module     │
│  - Connection pool   │
│  - query_db()        │
│  - execute_write_..()│
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   PostgreSQL DB      │
│   - users table      │
│   - refresh_tokens   │
└──────────────────────┘
```

## Testing

```bash
# Run auth tests
SKIP_DB_INIT=true CERTAMEN_SKIP_AUTH=true pytest tests/integration/test_gui_auth.py -v

# Run all integration tests
SKIP_DB_INIT=true pytest tests/integration/ -v
```

## Production Deployment

1. **Generate strong JWT secret**:

   ```bash
   openssl rand -base64 64
   ```

2. **Use environment variables** (never hardcode secrets):

   ```bash
   export CERTAMEN_JWT_SECRET="<generated-secret>"
   export CERTAMEN_DB_PASSWORD="<strong-password>"
   ```

3. **Enable SSL** for PostgreSQL connections in production

4. **Set up database backups** for users and refresh_tokens tables

5. **Monitor token usage** and set up cleanup jobs for expired tokens:

   ```sql
   DELETE FROM refresh_tokens
   WHERE expires_at < NOW() OR revoked_at IS NOT NULL;
   ```

6. **Never set `CERTAMEN_SKIP_AUTH=true`** in production

## Troubleshooting

**"Database pool is not initialized"**

- Make sure PostgreSQL is running
- Check `CERTAMEN_DB_*` environment variables
- Verify database exists: `psql -l | grep certamen`
- Run migration: `python scripts/db/init_auth_db.py`

**"Invalid token" errors**

- Check `CERTAMEN_JWT_SECRET` is set consistently
- Verify token hasn't expired (15 min lifetime)
- Use refresh token endpoint to get new access token

**"Connection pool error"**

- Check PostgreSQL connection limits
- Adjust `CERTAMEN_DB_POOL_MAX_SIZE` if needed
- Ensure connections are being returned to pool

**Tests failing with database errors**

- Set `SKIP_DB_INIT=true` for tests
- Use `CERTAMEN_SKIP_AUTH=true` for tests without database
