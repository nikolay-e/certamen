# Certamen Framework Architecture

## Naming & Structure Contract

This document defines the canonical naming conventions and directory structure for the Certamen ecosystem. All repositories must conform to these standards.

### Directory Naming Conventions

| Layer | Convention | Examples |
|-------|------------|----------|
| Python packages under `src/` | `snake_case` | `certamen_core`, `certamen` |
| Python modules | `snake_case.py` | `tournament.py`, `web_server.py` |
| TypeScript/frontend code | `kebab-case` or `PascalCase` | `use-websocket.ts`, `App.tsx` |
| Workflow templates | `kebab-case.yml` | `tournament-elimination.yml` |
| YAML extension | `.yml` only | Never `.yaml` |
| Top-level directories | `lowercase` | `frontend/`, `src/`, `docs/` |

### Canonical Directory Locations

| Concept | certamen-core | certamen-framework |
|---------|---------------|---------------------|
| Business logic | `src/certamen_core/domain/` | `src/certamen/` |
| Workflow nodes | `src/certamen_core/domain/workflow/nodes/` | Uses core nodes |
| LLM adapters | `src/certamen_core/adapters/llm/` | `src/certamen/models/` |
| Web server | N/A | `src/certamen/web/` |
| Frontend SPA | N/A | `frontend/` |
| CLI | `src/certamen_core/interfaces/cli/` | `src/certamen/cli/` |
| Configuration | `src/certamen_core/adapters/config/` | `src/certamen/config/` |
| Tests | `tests/integration/` | `tests/integration/` |

### Architecture Vocabulary

Both repositories use **hexagonal architecture** terminology:

| Term | Description | Core Location | Framework Location |
|------|-------------|---------------|-------------------|
| **Domain** | Core business logic | `domain/` | Uses core |
| **Ports** | Interface contracts | `ports/` | Uses core |
| **Adapters** | Implementation of ports | `adapters/` | `models/`, `config/` |
| **Application** | Orchestration services | `application/` | Uses core |
| **Interfaces** | External entry points | `interfaces/` | `cli/`, `web/` |
| **Support** | Cross-cutting utilities | `support/` | `utils/`, `logging/` |

### Public API Surface

Both packages export minimal public APIs via `__init__.py`:

```python
# certamen-core
from certamen_core import Certamen, __version__

# certamen-framework
from certamen import Certamen, __version__
```

All other imports are considered internal and may change without notice.

### Re-export Pattern

Use explicit re-exports with `__all__` for public API:

```python
# CORRECT: Explicit re-export
from .submodule import ClassName as ClassName

__all__ = ["ClassName"]
```

```python
# AVOID: Implicit re-export
from .submodule import ClassName  # Not recognized by type checkers
```

### Forbidden Patterns

1. **No unit tests** - Integration/E2E only
2. **No inline documentation** - Code must be self-documenting
3. **No duplicate module paths** - One canonical location per concept
4. **No symlinks in git** - Except for `.claude/` workspace inheritance
5. **No runtime artifacts in git** - Use `.gitignore` for `*.db`, `outputs/`, etc.

### Runtime Artifacts (Must be .gitignored)

```gitignore
# SQLite and runtime databases
*.db
*.db-wal
*.db-shm
*.db-journal

# Output directories
outputs/
results/
var/
.cache/

# Build artifacts
dist/
build/
*.egg-info/
```

---

## Component Architecture

### Web Mode Architecture

```
┌─────────────────────────────────────────────────────┐
│  Browser (User)                                     │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP :80 / WebSocket
                   ▼
┌──────────────────────────────────────────────────────┐
│  Frontend Container (Nginx)                          │
│  • Serves: React SPA from frontend/dist/             │
│  • Proxies: /api/*, /ws → backend:8765               │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  Backend Container (Python)                          │
│  • Location: src/certamen/web/server.py             │
│  • Class: WebServer                                  │
│  • WebSocket: /ws for real-time events               │
│  • Uses: certamen-core for tournament logic         │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  Database (PostgreSQL)                               │
│  • Purpose: Authentication only                      │
│  • Schema: scripts/db/init_auth_db.sql              │
└──────────────────────────────────────────────────────┘
```

### Package Dependencies

```
certamen (framework)
    ↓ depends on
certamen-core
    ├── Tournament orchestration
    ├── Workflow node system
    ├── Graph executor
    └── LLM adapters (LiteLLM)
```

### Key Classes

| Class | Location | Responsibility |
|-------|----------|----------------|
| `WebServer` | `src/certamen/web/server.py` | HTTP/WebSocket server |
| `AsyncExecutor` | `certamen-core` | Workflow graph execution |
| `ModelComparison` | `certamen-core` | Tournament orchestration |

---

## Guardrails & Enforcement

### Pre-commit Hooks

Both repositories use pre-commit with:

1. **import-linter** - Enforces layer dependencies
2. **ruff** - Linting and formatting
3. **mypy** - Type checking
4. **detect-secrets** - Prevents secret commits

### Import Rules

Enforced via `.importlinter`:

```ini
[importlinter:contract:layers]
type = layers
layers =
    certamen.web
    certamen.cli
    certamen.models
    certamen.config
    certamen.utils
```

### CI Checks

- All tests must pass
- No linting errors
- Type checks must pass
- No secrets in code
- Structure linter validates naming

---

## Vocabulary Mapping (Core ↔ Framework)

Both repositories share architecture concepts but use different directory names. This table maps equivalent concepts:

| Core (`certamen-core`) | Framework (`certamen-framework`) | Description |
|-------------------------|-----------------------------------|-------------|
| `domain/` | Uses core | Business logic |
| `ports/` | Uses core | Interface contracts |
| `adapters/` | `models/`, `config/` | Implementations |
| `application/` | Uses core | Orchestration |
| `interfaces/` | `cli/`, `web/` | Entry points |
| `support/` | `utils/`, `logging/` | Utilities |

**Note:** Framework depends on `certamen-core` for domain logic and extends it with web interface (`web/`) and additional configuration (`config/`).

---

## Migration Notes

### From gui/ to frontend/

The frontend directory was renamed from `gui/` to `frontend/` to avoid naming collision with `src/certamen/gui/` (now `src/certamen/web/`).

**Old paths:**

- `gui/` → TypeScript/React frontend
- `src/certamen/gui/` → Python WebSocket server

**New paths:**

- `frontend/` → TypeScript/React frontend
- `src/certamen/web/` → Python WebSocket server

### Class Renames

| Old Name | New Name |
|----------|----------|
| `GUIServer` | `WebServer` |
| `run_gui_server` | `run_web_server` |

### Environment Variables

| Old | New |
|-----|-----|
| `CERTAMEN_GUI_DIR` | `CERTAMEN_FRONTEND_DIR` |
