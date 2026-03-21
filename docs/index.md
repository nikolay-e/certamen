# Certamen Framework

AI tournament framework for decision synthesis through model competition and critique.

## Quick Start

```bash
# Install
pip install certamen

# Run tournament
certamen --config config.yml

# Start web interface
certamen gui --host 0.0.0.0 --port 8765
```

## Documentation

- [Getting Started](try-it-now.md) - Quick start guide
- [Authentication](authentication.md) - User authentication system
- [ROI Calculator](calculator.html) - Calculate ROI for your use case

## Architecture

Certamen operates in two modes:

### CLI Mode

Direct command-line execution for tournaments and workflows.

### Web Mode

Microservices architecture with:

- **Frontend** (`frontend/`) - React SPA with visual workflow editor
- **Backend** (`src/certamen/web/`) - Python aiohttp WebSocket server
- **Database** - PostgreSQL for authentication

## Source Code

- [GitHub Repository](https://github.com/nikolay-e/certamen-framework)
- [Full Documentation](../CLAUDE.md)
