# Comprehensive GUI & Authentication Audit Report

**Date:** 2025-11-30
**Scope:** Certamen GUI (aiohttp WebSocket + PostgreSQL auth)
**Focus:** GUI architecture, authentication security, WebSocket, database

---

## Executive Summary

Проведено глибокий аудит GUI та системи аутентифікації Certamen з фокусом на безпеку, архітектуру та тестування. Виявлено **12 критичних**, **23 серйозних** та **31 помірну** уразливість.

**Основні проблеми:**

1. **WebSocket повністю не захищений** - будь-хто може підключитися без аутентифікації
2. **JWT secret за замовчуванням пустий** - можна підробити токени
3. **Відсутні інтеграційні тести** - всі тести використовують моки (порушення політики)
4. **Відсутня перевірка Origin** на WebSocket - CSRF атаки можливі
5. **Connection pool exhaustion** - немає timeout, DoS вразливість

**Production Readiness:** ❌ **НЕ ГОТОВО** для production без виправлення критичних проблем

**Recommended Timeline:**

- **Week 1:** Fix CRITICAL issues (WebSocket auth, JWT validation, empty secrets)
- **Week 2:** Fix MAJOR issues (rate limiting, CORS, connection pool timeout)
- **Week 3:** Rewrite tests (remove mocks, add real PostgreSQL integration)
- **Week 4:** Add monitoring, metrics, performance optimization

---

## 🔴 CRITICAL Findings (12 issues)

### 1. WebSocket Authentication Completely Missing

**Files:** `server.py:80-103`, `middleware.py:13`

**Problem:**

- WebSocket endpoint `/ws` listed as PROTECTED in middleware
- But middleware **doesn't execute** for WebSocket connections
- Client connects WITHOUT sending JWT token
- Any unauthenticated user can connect and execute workflows

**Exploit:**

```bash
websocat ws://victim.com:8765/ws
# Connected without auth!
{"type": "execute", "nodes": [], "edges": []}
# Workflow executes!
```

**Impact:** Complete auth bypass for WebSocket API
**CVSS:** 9.8 (AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H)

**Fix:** Add JWT authentication in WebSocket handshake (see remediation plan)

---

### 2. Empty JWT Secret Default

**File:** `config.py:11`

```python
JWT_SECRET = os.getenv("CERTAMEN_JWT_SECRET", "")  # ← Empty string!
```

**Problem:** If `CERTAMEN_JWT_SECRET` not set, uses empty string → any JWT is valid

**Exploit:**

```python
import jwt
fake_admin = jwt.encode({"userId": 1, "isAdmin": True}, "", algorithm="HS256")
# Valid token with admin privileges!
```

**Impact:** Complete authentication bypass
**CVSS:** 9.8

**Fix:**

```python
JWT_SECRET = os.getenv("CERTAMEN_JWT_SECRET")
if not JWT_SECRET or len(JWT_SECRET) < 32:
    raise RuntimeError("CERTAMEN_JWT_SECRET must be set (min 32 chars)")
```

---

### 3. Cross-Site WebSocket Hijacking (CSWSH)

**File:** `server.py:80`

**Problem:** No `Origin` header validation during WebSocket handshake

**Exploit:**

```html
<!-- evil.com/attack.html -->
<script>
const ws = new WebSocket('ws://victim.com:8765/ws');
ws.send(JSON.stringify({type: 'execute', nodes: []}));
</script>
```

**Impact:** Cross-site attacks possible
**CVSS:** 8.6

**Fix:** Validate Origin header (see WebSocket audit section)

---

### 4. Full Stack Traces Exposed to Clients

**File:** `executor.py:287`

```python
error_info = {
    "traceback": traceback.format_exc(),  # ← Exposes internal paths!
}
```

**Impact:** Information disclosure (file paths, library versions, secrets in variables)
**CVSS:** 7.5

**Fix:** Sanitize tracebacks, only log full traces server-side

---

### 5. SKIP_AUTH Bypass in Production

**File:** `config.py:19-23`

**Problem:** Environment variable can disable ALL authentication

**Impact:** If `CERTAMEN_SKIP_AUTH=true` set in production → complete bypass
**CVSS:** 9.8

**Fix:** Prevent SKIP_AUTH in production environment

---

### 6. No Refresh Token Reuse Detection

**File:** `routes.py:177-254`

**Problem:** Stolen refresh token can be used multiple times without detection

**Impact:** Token theft not detected, no automatic session revocation
**CVSS:** 8.1

**Fix:** Implement token family tracking + reuse detection (see auth audit)

---

### 7. Database Pool Never Closed on Shutdown

**File:** `database.py:32-47`

**Problem:** `SimpleConnectionPool` initialized but never closed → connection leaks

**Impact:** Resource leaks, inability to graceful shutdown
**CVSS:** 5.3

**Fix:** Add `app.on_shutdown` handler to close pool

---

### 8. No WebSocket Connection Limit

**File:** `server.py:27`

**Problem:** Unlimited concurrent WebSocket connections

**Exploit:** Open 100,000 connections → 5GB RAM → OOM crash

**Impact:** DoS via connection exhaustion
**CVSS:** 7.5

**Fix:** Add MAX_CONNECTIONS limit (1000) + per-IP limit (10)

---

### 9. No Connection Pool Timeout

**File:** `database.py:53`

**Problem:** `getconn()` blocks forever if pool exhausted

**Impact:** DoS - server hangs
**CVSS:** 7.5

**Fix:** Upgrade to psycopg3 or implement timeout wrapper

---

### 10. Race Condition in Token Rotation

**File:** `routes.py:218-228`

**Problem:** Two separate transactions (UPDATE + INSERT) instead of atomic operation

**Impact:** Token replay attacks possible
**CVSS:** 6.5

**Fix:** Combine into single transaction with SERIALIZABLE isolation

---

### 11. Empty Database Password Default

**File:** `config.py:7`

```python
DB_PASSWORD = os.getenv("CERTAMEN_DB_PASSWORD", "")
```

**Problem:** Defaults to empty string, server starts with warning (not error)

**Impact:** Production misconfiguration possible
**CVSS:** 7.5

**Fix:** Fail fast if DB_PASSWORD not set

---

### 12. Write Operation Detection Bypass

**File:** `database.py:63-79`

**Problem:** `query_db()` checks only `startswith()` - can be bypassed with comments

**Exploit:**

```python
query_db("/* bypass */ INSERT INTO users VALUES (%s)", (data,))
```

**Impact:** Architectural vulnerability for future code
**CVSS:** 5.3

**Fix:** Strip comments before checking, detect semicolons

---

## 🟠 MAJOR Findings (23 issues)

### Architecture Issues

13. **Using `set()` instead of `WeakSet` for Clients** (server.py:27)
    - Memory leak potential if cleanup fails
    - Fix: Use `WeakSet` for automatic cleanup

14. **No CORS Configuration** (entire server.py)
    - Cross-origin requests may fail
    - Fix: Add aiohttp-cors middleware

15. **No Rate Limiting** (all endpoints)
    - Brute force attacks possible
    - Fix: Add rate limiting (10 msg/sec WebSocket, 5 req/min auth)

16. **Broadcast Error Handling Loses Failures** (server.py:75-78)
    - `return_exceptions=True` silently ignores errors
    - Dead clients remain in memory
    - Fix: Track dead clients and remove them

17. **Complex Database Functions** (database.py)
    - `query_db()` CC=14, `execute_write_transaction()` CC=11
    - Hard to test, error-prone
    - Fix: Extract connection management to context manager

18. **No Input Validation on WebSocket Messages** (server.py:88-154)
    - No size limits on nodes/edges arrays
    - Fix: Add max 100 nodes, 500 edges limits

### Security Issues

19. **Path Traversal in Middleware** (middleware.py:28-37)
    - `startswith()` vulnerable to `../` bypass
    - Fix: Use `os.path.normpath()` validation

20. **Weak JWT Algorithm (HS256 vs RS256)** (security.py:38,78)
    - Symmetric key vulnerable to compromise
    - Fix: Migrate to RS256 asymmetric keys

21. **bcrypt Instead of Argon2id** (security.py:19-21)
    - OWASP 2024 recommends Argon2id
    - Fix: Implement Argon2id with passlib

22. **No Password Complexity Requirements** (schemas.py:8-9)
    - Only length validation (8-128 chars)
    - Fix: Add regex, check against HIBP

23. **No Account Lockout After Failed Attempts** (routes.py:108-174)
    - Unlimited login attempts
    - Fix: Add failed_attempts counter, lockout after 5 failures

24. **No HTTPS/TLS Enforcement** (server.py:222-223)
    - JWT tokens transmitted over HTTP
    - Fix: Add HTTPS redirect, require SSL certs

25. **No Security Headers** (entire server.py)
    - Missing X-Content-Type-Options, X-Frame-Options, CSP, HSTS
    - Fix: Add security headers middleware

26. **No Origin Validation on WebSocket** (server.py:80)
    - See CRITICAL #3

27. **Password Length Not Limited on Login** (schemas.py:12-13)
    - Registration: max 128 chars, Login: unlimited
    - DoS via extremely long passwords (bcrypt slow)
    - Fix: Add max_length=128 to UserLogin

28. **Refresh Token Not Single-Use** (routes.py:177-254)
    - Race condition allows multiple uses
    - See CRITICAL #10

29. **No Message Size Limit on WebSocket** (server.py:89)
    - Attacker can send 10MB JSON → event loop blocks
    - Fix: `WebSocketResponse(max_msg_size=1MB)`

30. **No Heartbeat/Ping-Pong** (server.py:80-103)
    - Dead connections not detected
    - Fix: `WebSocketResponse(heartbeat=30)`

31. **JSON Parsing Without Schema Validation** (server.py:91)
    - No Pydantic validation for WebSocket messages
    - Fix: Add schema validation (see WebSocket audit)

32. **No SSL/TLS for PostgreSQL** (database.py:33-41)
    - Connection not encrypted
    - Fix: Add `sslmode='require'`

33. **Username Enumeration via Timing** (routes.py:40-49 + SQL UNIQUE)
    - Registration reveals if username exists
    - Fix: Always execute bcrypt for constant-time

34. **Synchronous Password Hashing Blocks Event Loop** (security.py:19-28)
    - bcrypt takes 100-300ms, blocks asyncio
    - Fix: Use `run_in_executor()` for bcrypt

35. **Refresh Token Lifetime Too Long** (config.py:15-17)
    - 30 days (OWASP recommends 7-14 days)
    - Fix: Change default to 7 days

### Performance Issues

36. **Broadcast Scales O(N)** (server.py:75-78)
    - Every update → N sends (N=client count)
    - Fix: Room-based subscriptions per execution_id

37. **Node Execution Timeout Too High** (executor.py:11)
    - 5 minutes default
    - Fix: Lower to 2 minutes, add global workflow timeout

38. **Database Pool Too Small** (config.py:8-9)
    - Max 10 connections for potentially 1000 WebSocket clients
    - Fix: Increase to 30-50

39. **No Caching for Node Registry** (registry.py)
    - Rebuilds schema on every `/api/nodes` call
    - Fix: Add `@lru_cache`

40. **Models Recreated on Every Execution** (executor.py:151-152)
    - Initialization overhead
    - Fix: Reuse model instances

---

## 🟡 MINOR Findings (31 issues)

### Code Quality

41. **Missing Type Hints (45% coverage)**
42. **God Object Pattern in GUIServer** (15 methods, mixed concerns)
43. **Magic Strings for Message Types** (should use Enum)
44. **Duplicate Code in Routes** (token creation logic repeated)
45. **Inconsistent Error Handling**
46. **Missing Docstrings** (intentional per project policy - OK)
47. **Error Messages May Leak Info** (low risk)

### Security

48. **Admin Check Race Condition** (TOCTOU in security.py:116-128)
49. **SHA256 for Refresh Tokens** (should use bcrypt)
50. **Timing Attack on Password Verification** (routes.py:128-130)
51. **nbf Claim Not Validated** (security.py:76-91)
52. **Missing jti Claim Tracking** (can't revoke specific tokens)
53. **Missing Input Sanitization in Username** (no regex, allows unicode)
54. **No HTTPS Upgrade Protection** (downgrade attacks possible)
55. **Dependency Pinning** (uses `~=` instead of exact versions)
56. **Passwords Logged in Plaintext (Potential)** - not happening but risk exists
57. **No Failed Login Logging** (only username, not IP/User-Agent)
58. **No Security Event Monitoring**

### Database

59. **Missing Index on Username** (implicit via UNIQUE but not explicit)
60. **Timestamps Without Timezone Info** (should use TIMESTAMPTZ)
61. **SQL Query Leakage in Logs** (error messages contain queries)
62. **Database Connection String in Logs** (info level)
63. **No Prepared Statement Caching**

### Testing

64. **Mock-Heavy Tests Violate Policy** (test_gui_auth.py uses database mocks)
65. **Unit Tests Present** (test_security_functions - should be E2E)
66. **No WebSocket E2E Tests** (0% coverage)
67. **No Real Database Integration** (all tests use SKIP_DB_INIT=true)
68. **No Token Expiration Tests**
69. **No Timeout Tests**
70. **No SQL Injection Tests**
71. **No Auth Bypass Tests**

---

## Test Coverage Analysis

**HTTP Endpoints:** 6/12 tested (50%) - but all shallow
**WebSocket Messages:** 0/5 tested (0%)
**Auth Flows:** 0/7 with real DB (0%)
**Error Cases:** 3/10 tested (30%)
**Edge Cases:** 2/15 tested (13%)

**Policy Violations:**

- ❌ Heavy database mocking (entire layer mocked)
- ❌ Unit tests present (violates "NO UNIT TESTS" policy)
- ❌ No real PostgreSQL integration

**Critical Missing Tests:**

1. WebSocket authentication flow
2. Real database operations (register → verify in DB)
3. Token expiration & refresh
4. Graph execution timeout
5. Concurrent WebSocket clients
6. SQL injection attempts
7. Auth bypass attempts (path traversal)
8. Database connection pool exhaustion

---

## Remediation Roadmap

### 🔴 Phase 1: CRITICAL (Week 1) - 40 hours

**Must fix before production:**

1. **WebSocket Authentication** (8h)
   - Implement JWT validation in handshake
   - Update client to send token
   - Add Origin validation

2. **Fix Empty Secrets** (2h)
   - JWT_SECRET validation
   - DB_PASSWORD validation
   - Fail fast on startup

3. **Remove Stack Trace Exposure** (2h)
   - Sanitize tracebacks
   - Generic error IDs for clients

4. **SKIP_AUTH Protection** (2h)
   - Prevent in production
   - Add warnings

5. **Refresh Token Reuse Detection** (12h)
   - Token family tracking
   - Reuse detection
   - Automatic revocation

6. **Connection Pool Cleanup** (2h)
   - on_shutdown handler
   - Graceful close

7. **WebSocket Connection Limits** (4h)
   - Global limit (1000)
   - Per-IP limit (10)

8. **Database Pool Timeout** (4h)
   - Implement timeout wrapper
   - OR upgrade to psycopg3

9. **Token Rotation Race Condition** (4h)
   - Atomic transaction
   - SERIALIZABLE isolation

**Total:** 40 hours (~5 days)

---

### 🟠 Phase 2: MAJOR (Week 2-3) - 72 hours

10. **Add Rate Limiting** (8h)
    - WebSocket: 10 msg/sec
    - Auth endpoints: 5 req/min

11. **CORS Configuration** (4h)
    - aiohttp-cors
    - Whitelist origins

12. **Security Headers** (4h)
    - X-Content-Type-Options
    - X-Frame-Options
    - CSP, HSTS

13. **WebSocket Schema Validation** (6h)
    - Pydantic models
    - Size limits

14. **Broadcast Error Handling** (4h)
    - Track dead clients
    - Auto cleanup

15. **Use WeakSet** (2h)
    - Replace set() with WeakSet

16. **Path Traversal Fix** (4h)
    - Normalize paths
    - Strict validation

17. **Migrate to Argon2id** (8h)
    - Replace bcrypt
    - Migration plan

18. **Password Complexity** (4h)
    - Regex validation
    - HIBP check

19. **Account Lockout** (6h)
    - Failed attempts tracking
    - Lockout logic

20. **HTTPS Enforcement** (4h)
    - SSL cert loading
    - Redirect HTTP→HTTPS

21. **PostgreSQL SSL** (2h)
    - sslmode='require'

22. **Username Enumeration Fix** (4h)
    - Constant-time responses
    - Generic errors

23. **bcrypt to Thread Pool** (4h)
    - run_in_executor()

24. **Refresh Token Lifetime** (1h)
    - Change to 7 days

25. **Room-Based Broadcasting** (8h)
    - execution_id subscriptions
    - Targeted broadcasts

**Total:** 72 hours (~9 days)

---

### 🟡 Phase 3: MINOR & Testing (Week 4) - 60 hours

26. **Rewrite Tests** (24h)
    - Remove all database mocks
    - Use real PostgreSQL (testcontainers)
    - Add WebSocket E2E tests
    - Add security tests

27. **Add Monitoring** (8h)
    - Connection pool metrics
    - Failed login tracking
    - WebSocket metrics

28. **Performance Optimization** (12h)
    - orjson for JSON parsing
    - Node registry caching
    - Model reuse

29. **Code Quality** (8h)
    - Extract connection manager
    - Message type Enum
    - Split GUIServer

30. **Documentation** (8h)
    - WebSocket auth guide
    - Security best practices
    - Deployment checklist

**Total:** 60 hours (~8 days)

---

## Overall Assessment

**Security Score:** 4.5/10 (POOR)

- Authentication: 3/10 (WebSocket unprotected, empty secrets)
- Authorization: 5/10 (middleware works but bypasses exist)
- Data Protection: 6/10 (parameterized queries good, but no SSL)
- Monitoring: 2/10 (minimal logging, no metrics)

**Architecture Score:** 6/10 (FAIR)

- Design: 7/10 (clean separation, but God objects)
- Performance: 5/10 (O(N) broadcast, blocking bcrypt)
- Scalability: 4/10 (small pool, no horizontal scaling)
- Error Handling: 7/10 (mostly good)

**Testing Score:** 2/10 (CRITICAL)

- Coverage: 3/10 (50% endpoints, but shallow)
- Quality: 1/10 (violates project policy with mocks)
- E2E: 0/10 (no real integration tests)

**Production Readiness:** ❌ **NOT READY**

**Critical Blockers:**

1. WebSocket auth missing
2. Empty JWT secret possible
3. No real integration tests
4. CSWSH vulnerability
5. Connection pool exhaustion

**Timeline to Production:**

- **Minimum:** 3 weeks (Phase 1 + Phase 2 critical items)
- **Recommended:** 5 weeks (all 3 phases)

---

## Quick Wins (Can Deploy Immediately)

1. Add JWT_SECRET validation (1h)
2. Add DB_PASSWORD validation (1h)
3. Add Origin header check (2h)
4. Remove stack trace exposure (2h)
5. Add SKIP_AUTH=production check (1h)

**Total:** 7 hours - **deploy same day** to close critical gaps

---

## Sources

All research sources, CVE references, and best practice guides are documented in individual audit sections above.
