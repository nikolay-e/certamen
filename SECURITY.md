# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.0.x   | :white_check_mark: |

## Reporting a Vulnerability

The Arbitrium Framework team takes security seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, use GitHub's **Private Vulnerability Reporting** (recommended):

1. Go to the [Security tab](https://github.com/nikolay-e/arbitrium-framework/security)
2. Click "Report a vulnerability"
3. Fill out the form with details

This is the fastest and most secure way to reach the maintainers.

### What to Include

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

- **Acknowledgment:** We'll acknowledge receipt of your vulnerability report within 48 hours
- **Communication:** We'll keep you informed about our progress toward a fix
- **Credit:** We'll publicly credit you for the discovery (unless you prefer to remain anonymous)
- **Timeline:** We aim to issue a fix within 90 days of report

### Responsible Disclosure

We ask that you:

- Give us reasonable time to fix the vulnerability before any public disclosure
- Make a good faith effort to avoid privacy violations, data destruction, or service interruption
- Do not exploit the vulnerability beyond the proof of concept

### Bug Bounty

We currently do not offer a paid bug bounty program. However, we deeply appreciate security reports and will:

- Publicly credit your contribution
- Fast-track your pull requests
- Provide swag/merch when available

## Security Considerations for Users

### API Keys and Secrets

**NEVER commit API keys to version control.** Arbitrium Framework supports multiple secure methods for managing secrets:

1. **Environment variables** (recommended):

   ```bash
   export OPENAI_API_KEY="sk-..."  # pragma: allowlist secret
   export ANTHROPIC_API_KEY="sk-ant-..."  # pragma: allowlist secret
   ```

2. **Custom config.yml** (never commit!):

   ```yaml
   # config.yml (add to .gitignore)
   # Override api_providers.yml defaults
   secrets:
     providers:
       openai: "sk-..."
       anthropic: "sk-ant-..."
   ```

3. **1Password integration** (enterprise):

   ```yaml
   secrets:
     source: "1password"
     vault: "Arbitrium Framework"
   ```

### Model Provider Security

When using external LLM providers:

- **Data Privacy:** Your prompts and responses are sent to third-party APIs
- **Data Retention:** Check each provider's data retention policy
- **Compliance:** Ensure provider meets your regulatory requirements (GDPR, HIPAA, etc.)
- **Rate Limiting:** Implement rate limits to prevent abuse

### Secure Configuration

In production environments:

```yaml
# config.yml
features:
  save_reports_to_disk: false  # Avoid writing sensitive data to disk
  llm_compression: true         # Reduce data sent to providers

retry:
  max_attempts: 3               # Limit retry abuse
```

### Dependencies

Arbitrium Framework uses `pip-audit` and `safety` in CI/CD to scan for known vulnerabilities. To check your installation:

```bash
pip install pip-audit
pip-audit --desc

# Or use safety
pip install safety
safety check
```

### Network Security

If running in a restricted environment:

- **Proxy Support:** Set `HTTP_PROXY` and `HTTPS_PROXY` environment variables
- **Firewall Rules:** Allow outbound HTTPS to provider APIs
- **TLS/SSL:** All API calls use TLS 1.2+ by default

## Known Limitations

### Not Designed For

Arbitrium Framework is **not designed** for:

- ❌ **Adversarial Use:** Do not use for generating malware, phishing, or other malicious content
- ❌ **High-Security Data:** Not suitable for classified or highly sensitive information without additional controls
- ❌ **Real-Time Safety-Critical Systems:** No SLA guarantees, not designed for life-safety applications
- ❌ **Compliance-Heavy Environments:** No built-in HIPAA/SOC2 compliance features (consider "Arbitrium Enterprise" roadmap)

### Designed For

Arbitrium Framework is designed for:

- ✅ **Strategic Decision-Making:** Business strategy, technical architecture, research planning
- ✅ **Synthesis:** Combining perspectives from multiple models
- ✅ **Auditable Decisions:** Full provenance tracking and cost accounting
- ✅ **Research:** Academic experiments, benchmarking, model evaluation

## Security Updates

Security updates are released as patch versions (0.0.x). Subscribe to:

- **GitHub Watch:** Click "Watch" → "Releases only" on [our repo](https://github.com/nikolay-e/arbitrium-framework)
- **Security Advisories:** [GitHub Security Advisories](https://github.com/nikolay-e/arbitrium-framework/security/advisories)

## Questions?

For security-related questions (not vulnerabilities), please:

- Open a [GitHub Discussion](https://github.com/nikolay-e/arbitrium-framework/discussions) (preferred)
- Or file a regular [GitHub Issue](https://github.com/nikolay-e/arbitrium-framework/issues) with the `security-question` label

For general questions, use [GitHub Discussions](https://github.com/nikolay-e/arbitrium-framework/discussions).

---

**Last Updated:** October 2025
