# Threatforge ML Server — V3

> **Laptop 2 is private. Do NOT expose port 8001 publicly.**
>
> It communicates exclusively with Laptop 1 through the Tailscale private network.

---

## Architecture

```
Browser
  ↓
 Gateway :8000  (public-facing)
  ↓  Tailscale private network
 ML Server :8001  ← THIS SERVER
  ↓
Multi-stage security analysis
  ↓
Structured response
  ↓
Dashboard
```

---

## Datasets and ML Models

> **Note:** To keep the repository lightweight, large files such as datasets (`data/`, `raw_data/`) and ML models (`models/`) are **not** included in this Git repository.
> 
> Only the source code and folder structures are uploaded. If you need access to the datasets or the pre-trained ML models to run the server, you must request them directly from the project owner and place them in their respective directories.

---

## V3 Analysis Pipeline

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | URL Normalization | Canonical form, scheme/host lowercase, default port stripping |
| 1b | URL Validation | Scheme, hostname, port, encoding checks |
| 1c | URL Manipulation | @ abuse, IP URL, encoded chars, embedded URLs |
| 2 | Domain Analysis | Entropy, subdomains, punycode, suspicious TLD |
| 2b | Spoofing Detection | Levenshtein typosquatting, homoglyphs, brand registry |
| 3 | DNS Resolution | SSRF-validated IP checking, private IP detection |
| 4 | SSL/TLS Analysis | Certificate validity, expiry, hostname match |
| 5 | Reachability | SSRF-protected HTTP fetch with redirect following |
| 5b | Redirect Analysis | Cross-domain, TLD change, HTTPS→HTTP downgrade |
| 5c | HTML Analysis | Login forms, credential fields, external form action |
| 5d | Fake Login Detection | Combined signals for phishing page patterns |
| 6 | XGBoost ML | URL-feature-based phishing probability |
| 7 | Evidence Collection | Structured evidence from all stages |
| 7b | Threat Intelligence | Optional VirusTotal / future providers |
| 8 | Risk Engine | Multi-signal weighted scoring |
| 9 | Explanation Engine | Deterministic human-readable explanation |
| 9b | LLM Enhancement | Optional local Qwen3 prose improvement |

### Key Principles

- **UNREACHABLE ≠ SAFE** — A site that can't be reached may still be phishing.
- **Valid SSL ≠ Legitimate** — Phishing sites routinely obtain free certificates.
- **LLM ≠ Security Decision Maker** — The LLM only explains; the deterministic engine decides.
- **No Fabrication** — Unchecked features return `not_checked` / `not_configured`, never `safe`.
- **SSRF Protection** — Every outbound request is validated before execution.

---

## Directory Structure

```
ml-server/
│
├── app/
│   ├── main.py              — FastAPI application + all endpoints
│   ├── config.py            — Pydantic settings (env var loading)
│   ├── security.py          — API key authentication
│   ├── model_registry.py    — Model loading at startup
│   ├── risk.py              — V1 legacy + V3 multi-evidence risk engine
│   ├── rules.py             — Static URL rule detections (legacy)
│   ├── evidence.py          — Evidence collection + weighting
│   ├── explanation.py       — Deterministic explanation engine
│   │
│   ├── models/
│   │   └── phishing.py      — XGBoost model wrapper
│   │
│   ├── features/
│   │   ├── url.py           — URL features + normalization + validation
│   │   ├── domain.py        — Domain structural analysis
│   │   ├── dns.py           — DNS resolution + SSRF IP validation
│   │   └── ssl.py           — SSL/TLS certificate analysis
│   │
│   ├── analyzers/
│   │   ├── ssrf.py          — SSRF protection (authoritative guard)
│   │   ├── reachability.py  — SSRF-protected HTTP fetcher
│   │   ├── redirect.py      — Redirect chain analysis
│   │   ├── spoofing.py      — Brand look-alike detection
│   │   ├── website.py       — HTML analysis (no JS execution)
│   │   ├── login_page.py    — Fake login page detection
│   │   └── ssl.py           — SSL analyzer wrapper
│   │
│   └── intelligence/
│       ├── brands.py        — Brand registry + typosquatting detection
│       └── threat_intel.py  — Threat intel provider interface
│
├── llm/
│   └── provider.py          — LLM explanation provider (Qwen3 / Null)
│
├── models/
│   ├── registry.json        — Model registry configuration
│   └── ml-phishing/         — Trained XGBoost model artifacts
│
├── pipeline/                — Offline training data pipeline
├── raw_data/                — Raw training datasets (offline)
├── data/                    — Generated training data (offline)
├── train.py                 — Training script (offline)
├── run_pipeline.py          — Pipeline runner (offline)
├── tests/                   — Test suite
├── requirements.txt
├── .env                     — Local secrets (NOT in git)
└── .env.example             — Template for .env
```

---

## Installation

Python 3.12.x recommended.

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration

Copy `.env.example` to `.env` and set your values:

```env
ML_API_KEY=replace-with-a-strong-random-secret
HOST=0.0.0.0
PORT=8001

MODEL_REGISTRY_FILE=models/registry.json
DEFAULT_MODEL=phishing

# Timeouts
CONNECT_TIMEOUT=5.0
READ_TIMEOUT=10.0
REQUEST_TIMEOUT=20.0
MAX_REDIRECTS=5
MAX_RESPONSE_BYTES=5242880
MAX_HTML_BYTES=2097152

# Optional threat intelligence
THREAT_INTEL_PROVIDER=none
VIRUSTOTAL_API_KEY=

# Optional LLM explanation
LLM_ENABLED=false
LLM_HOST=127.0.0.1
LLM_PORT=8002
LLM_MODEL=qwen3:8b
```

> The same `ML_API_KEY` must be set on Laptop 1.

---

## Start the Server

```powershell
# Windows
.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Check it's running:

```powershell
curl http://127.0.0.1:8001/health/live
```

Expected:

```json
{"status": "ok", "service": "cyberguard-ml-server"}
```

---

## API Endpoints

### Public (no auth)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health/live` | Liveness probe |
| GET | `/health` | Health + loaded models |

### Authenticated (X-API-Key header)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/models` | List loaded models |
| POST | `/predict` | **Legacy** — XGBoost static analysis |
| POST | `/api/v1/models/{name}/predict` | **Legacy** — Model-specific prediction |
| POST | `/api/v1/analyze-url` | **V3** — Full pipeline analysis |

### Authentication

All authenticated requests require:

```http
X-API-Key: <ML_API_KEY value from .env>
```

---

## V3 API Usage

### Request

```http
POST /api/v1/analyze-url
X-API-Key: <secret>
Content-Type: application/json

{
  "url": "https://example.com"
}
```

### Response (abbreviated)

```json
{
  "url": "https://example.com",
  "normalized_url": "https://example.com",
  "website_status": "live",
  "verdict": "SAFE",
  "risk_score": 4.5,
  "severity": "low",
  "ml_probability": 0.031,
  "model_version": "phishing-v1",
  "detections": [],
  "evidence": [],
  "domain": {
    "hostname": "example.com",
    "registered_domain": "example.com",
    "tld": "com",
    "has_suspicious_tld": false
  },
  "dns": {
    "status": "resolved",
    "resolved_ips": ["93.184.216.34"],
    "has_private_ip": false
  },
  "ssl": {
    "status": "checked",
    "cert_valid": true,
    "hostname_match": true
  },
  "redirects": {
    "redirect_count": 0,
    "domain_changed": false
  },
  "website": {
    "has_login_form": false,
    "has_external_form_action": false
  },
  "threat_intel": {
    "url": {"status": "not_configured"},
    "domain": {"status": "not_configured"}
  },
  "explanation": {
    "summary": "No significant phishing indicators were detected...",
    "reasons": [],
    "recommendation": ["Always verify you are on the correct domain before logging in."],
    "certainty": "medium"
  }
}
```

### Possible Verdicts

| Verdict | Meaning |
|---------|---------|
| `INVALID_URL` | URL failed validation |
| `UNREACHABLE` | Site could not be reached (≠ safe) |
| `SAFE` | No significant signals detected |
| `SUSPICIOUS` | Moderate evidence — exercise caution |
| `PHISHING` | Strong evidence of phishing |
| `ANALYSIS_ERROR` | Unexpected error during analysis |

---

## SSRF Protection

The live website fetcher has mandatory SSRF protection.

**Blocked destinations:**

- `127.0.0.0/8` — loopback
- `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` — private RFC1918
- `169.254.0.0/16` — link-local + AWS/GCP/Azure metadata (169.254.169.254)
- `100.64.0.0/10` — carrier-grade NAT / Tailscale range
- `::1` — IPv6 loopback
- `fc00::/7` — IPv6 unique-local
- `fe80::/10` — IPv6 link-local
- `localhost`, `*.local`, `*.internal`, `*.localhost`
- Single-label hostnames (`intranet`)
- Non-http(s) schemes

**Every redirect destination is also re-validated.**

A redirect from a public URL to `http://127.0.0.1:8001` is blocked.

---

## Security Model

- ML server is private — not publicly routable.
- Authentication: `X-API-Key` on every sensitive endpoint.
- API keys are never logged or included in responses.
- The LLM is separate (port 8002) and cannot make security decisions.
- The risk engine is deterministic — the LLM only explains.

---

## Getway Integration

Getway must set in its `.env`:

```env
ML_URL=http://100.x.x.x:8001    # Tailscale IP of MLserver ( if not public) 
ML_API_KEY=<same secret as ML server>
```

Getway calls:

- Legacy: `POST /predict` (backward compatible)
- V3: `POST /api/v1/analyze-url`

---

## Running Tests

```powershell
.venv\Scripts\Activate.ps1
python -m pytest tests/ -v
```

### Test coverage

- `tests/test_url.py` — Normalization, validation, manipulation detection
- `tests/test_ssrf.py` — SSRF blocking (127.0.0.1, localhost, private IPs, etc.)
- `tests/test_spoofing.py` — Brand look-alikes (paypa1, micr0soft, goog1e)
- `tests/test_risk.py` — Risk engine (legacy + V3), unreachable ≠ safe
- `tests/test_html.py` — Login form, external action, iframe detection
- `tests/test_api.py` — Endpoints, auth, SSRF via API

---

## Training Pipeline (Offline)

The training pipeline is completely separate from the live analyzer.
Live URL fetching is never mixed into training.

```powershell
python run_pipeline.py   # ingest → clean → feature extraction
python train.py          # train XGBoost, save to models/ml-phishing/
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model 'phishing' is not loaded` | Run `python train.py` to build the model |
| `ML_API_KEY is not configured` | Set `ML_API_KEY=...` in `.env` |
| `403 Forbidden` | API key mismatch between Laptop 1 and Laptop 2 |
| `ssrf_blocked` verdict | URL points to private/internal network — expected |
| `UNREACHABLE` verdict | Domain doesn't resolve or host is down |
| LLM not working | Check `LLM_ENABLED`, `LLM_HOST`, `LLM_PORT`, `LLM_MODEL` |

---

## What's New in V3

- Full website intelligence pipeline (reachability, SSL, redirect, HTML)
- SSRF protection on all outbound requests
- Brand impersonation detection (Levenshtein + homoglyphs)
- Multi-evidence risk engine (not just ML probability)
- Deterministic explanation engine
- Optional LLM prose enhancement (Qwen3 8B)
- Optional threat intelligence (VirusTotal)
- Explicit INVALID_URL / UNREACHABLE / ANALYSIS_ERROR verdicts
- 112 automated tests
