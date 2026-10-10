"""
CyberGuard ML Server — V3 main application.

Backward-compatible endpoints (DO NOT REMOVE — Laptop 1 uses these):
  GET  /health/live
  GET  /health
  GET  /models
  POST /predict
  POST /api/v1/models/{model_name}/predict

New V3 endpoint:
  POST /api/v1/analyze-url

Deepfake Detection endpoints (added in V3.1):
  POST /api/v1/deepfake/image
  POST /api/v1/deepfake/video
  POST /api/v1/deepfake/audio
  POST /api/v1/deepfake/video-call
  POST /api/v1/deepfake/multimedia

Architecture:
  Browser → Laptop 1 Gateway :8000 → Tailscale → Laptop 2 ML Server :8001

Laptop 2 is private. Never expose port 8001 publicly.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.model_registry import ModelRegistry
from app.risk import calculate_risk, calculate_risk_v3
from app.rules import detect_url_rules
from app.security import require_api_key
from app.features.url import (
    extract_url_features,
    normalize_url,
    validate_url,
    detect_url_manipulation,
)
from app.features.domain import analyse_domain
from app.features.dns import resolve_hostname
from app.features.ssl import analyse_ssl
from app.analyzers.ssrf import SSRFViolation, validate_url_for_fetch
from app.analyzers.reachability import fetch_url
from app.analyzers.redirect import analyse_redirects
from app.analyzers.spoofing import analyse_spoofing
from app.analyzers.ssl import analyse_ssl_for_url
from app.analyzers.website import analyse_website_html
from app.analyzers.login_page import analyse_login_page
from app.intelligence.threat_intel import threat_intel
from app.evidence import collect as collect_evidence, EvidenceItem
from app.explanation import build_explanation
from app.risk import calculate_risk_v3

logger = logging.getLogger(__name__)

# ── Logging: redact sensitive headers ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ── Model registry ────────────────────────────────────────────────────────────
registry = ModelRegistry(settings.model_registry_file)


@asynccontextmanager
async def lifespan(app):
    registry.load()
    logger.info(
        "Model registry loaded. Models: %s", registry.names()
    )

    # Initialize deepfake inference concurrency semaphore
    from app.deepfake.models.base import init_inference_semaphore
    init_inference_semaphore(settings.deepfake_max_concurrent_inferences)

    # Initialize deepfake service coordinator
    from app.deepfake.service import DeepfakeService
    app.state.deepfake_service = DeepfakeService(registry)
    logger.info("DeepfakeService initialized.")

    yield


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="CyberGuard ML Server",
    version="3.1.0",
    description=(
        "Private ML/Security platform for CyberGuard. "
        "Laptop 2 — not publicly exposed. "
        "Supports phishing detection and deepfake image detection."
    ),
    lifespan=lifespan,
    # Disable docs in production — uncomment for development.
    # docs_url=None,
    # redoc_url=None,
)

# ── CORS — allow any frontend origin to POST files ────────────────────────────
_cors_origins = [
    o.strip() for o in settings.cors_origins.split(",") if o.strip()
] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register feature routers ─────────────────────────────────────────────────
from app.deepfake.router import router as deepfake_router
app.include_router(deepfake_router)


# ── Request/response models ───────────────────────────────────────────────────

class PredictRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4096)


class AnalyzeURLRequest(BaseModel):
    url: str = Field(min_length=1, max_length=settings.max_url_length)


# ── Legacy URL validation (used by V1/V2 endpoints) ──────────────────────────

def _validate_url_legacy(url: str):
    """
    Minimal validation for backward-compatible endpoints.
    Raises HTTPException on failure.
    """
    parsed = urlparse(url.strip())

    if parsed.scheme.lower() not in {"http", "https"}:
        raise HTTPException(
            status_code=422,
            detail="Only HTTP and HTTPS URLs are supported.",
        )

    if not parsed.hostname:
        raise HTTPException(
            status_code=422,
            detail="URL hostname is required.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD-COMPATIBLE ENDPOINTS — DO NOT MODIFY RESPONSE SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health/live")
def health_live():
    return {
        "status": "ok",
        "service": "cyberguard-ml-server",
    }


@app.get("/health")
async def health():
    """
    Health check endpoint.

    ML status  — checks whether the XGBoost model registry is loaded.
    LLM status — queries Ollama GET /api/tags and verifies the configured
                 model is present. Does NOT call /api/generate (no GPU usage).
    """
    # ── ML status ──────────────────────────────────────────────────────────
    ml_status = "online" if registry.names() else "offline"

    # ── LLM status — fast Ollama reachability + model list check ──────────
    llm_status = "offline"
    ollama_base = f"http://{settings.llm_host}:{settings.llm_port}"

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollama_base}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                model_names = [m.get("name", "") for m in data.get("models", [])]
                # Match both "qwen3:4b" and "qwen3:4b-..." variants
                configured = settings.llm_model
                if any(name == configured or name.startswith(configured.split(":")[0] + ":")
                       and configured in name
                       for name in model_names):
                    llm_status = "online"
                # Exact prefix match: "qwen3:4b" in "qwen3:4b-q4_0" etc.
                if llm_status != "online":
                    for name in model_names:
                        if name == configured or name.startswith(configured):
                            llm_status = "online"
                            break
    except Exception:
        llm_status = "offline"

    return {
        "status": "ok",
        "service": "cyberguard-ml-server",
        "version": "3.0.0",
        "ml": {
            "status": ml_status,
            "loaded_models": registry.names(),
            "default_model": settings.default_model,
        },
        "llm": {
            "status": llm_status,
            "provider": "ollama",
            "model": settings.llm_model,
        },
    }


@app.get(
    "/models",
    dependencies=[Depends(require_api_key)],
)
def list_models():
    return {"models": registry.names()}


@app.post(
    "/predict",
    dependencies=[Depends(require_api_key)],
)
def predict(payload: PredictRequest):
    """
    Legacy endpoint — preserved for Laptop 1 backward compatibility.
    Uses V1/V2 risk engine only (static URL features + XGBoost).
    """
    _validate_url_legacy(payload.url)

    model = registry.get(settings.default_model)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model '{settings.default_model}' is not loaded."
            ),
        )

    features = extract_url_features(payload.url)
    detections = detect_url_rules(features)
    probability = model.predict_probability(features)
    risk_score, verdict = calculate_risk(probability, detections)

    return {
        "verdict": verdict,
        "risk_score": risk_score,
        "ml_probability": round(probability, 6),
        "model_version": f"{model.name}-{model.version}",
        "detections": detections,
        "features": features,
    }


@app.post(
    "/api/v1/models/{model_name}/predict",
    dependencies=[Depends(require_api_key)],
)
def predict_model(model_name: str, payload: PredictRequest):
    """
    Legacy model-specific endpoint — preserved for Laptop 1 backward compatibility.
    """
    _validate_url_legacy(payload.url)

    model = registry.get(model_name)

    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' is not loaded.",
        )

    features = extract_url_features(payload.url)
    detections = detect_url_rules(features)
    probability = model.predict_probability(features)
    risk_score, verdict = calculate_risk(probability, detections)

    return {
        "verdict": verdict,
        "risk_score": risk_score,
        "ml_probability": round(probability, 6),
        "model_version": f"{model.name}-{model.version}",
        "detections": detections,
        "features": features,
    }


from fastapi.responses import StreamingResponse

# ═══════════════════════════════════════════════════════════════════════════════
# V3 ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/v1/analyze-url/stream",
    dependencies=[Depends(require_api_key)],
)
async def analyze_url_stream(payload: AnalyzeURLRequest):
    from app.streaming import analyze_url_stream_generator
    model = registry.get(settings.default_model)
    return StreamingResponse(
        analyze_url_stream_generator(payload.url, model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no" # For Nginx etc
        }
    )

@app.post(
    "/api/v1/analyze-url",
    dependencies=[Depends(require_api_key)],
)
async def analyze_url(payload: AnalyzeURLRequest):
    """
    V3 full URL analysis endpoint.

    Combines:
    - URL normalization + validation
    - URL manipulation detection
    - Domain analysis
    - DNS resolution + SSRF IP validation
    - SSL/TLS analysis
    - Website reachability (SSRF-protected)
    - Redirect chain analysis
    - HTML analysis (login form, credential fields, etc.)
    - Fake login page detection
    - Brand spoofing / look-alike detection
    - Threat intelligence (if configured)
    - XGBoost ML probability
    - V3 multi-evidence risk engine
    - Deterministic explanation (+optional LLM enhancement)
    """
    raw_url = payload.url

    # ── Stage 1: URL normalisation & validation ───────────────────────
    normalized_url = normalize_url(raw_url)
    validation = validate_url(raw_url, max_length=settings.max_url_length)

    if not validation.valid:
        return {
            "url": raw_url,
            "normalized_url": normalized_url,
            "website_status": "invalid",
            "verdict": "INVALID_URL",
            "risk_score": 0.0,
            "ml_probability": None,
            "detections": [],
            "evidence": [],
            "domain": {},
            "dns": {},
            "ssl": {},
            "redirects": {},
            "website": {},
            "explanation": {
                "summary": (
                    f"The URL is invalid: {validation.detail}. "
                    "It cannot be analysed."
                ),
                "reasons": [validation.detail],
                "recommendation": ["Verify the URL is correct."],
                "certainty": "high",
            },
            "recommendation": ["Verify the URL is correct."],
            "validation": validation.as_dict(),
        }

    hostname = validation.hostname

    # ── Stage 1b: URL manipulation signals ───────────────────────────
    manipulation_signals = detect_url_manipulation(raw_url)

    # ── Stage 2: Domain analysis ─────────────────────────────────────
    domain_analysis = analyse_domain(normalized_url)

    # ── Stage 2b: Spoofing / look-alike detection ─────────────────────
    spoofing_analysis = analyse_spoofing(normalized_url)

    # ── Stage 3: DNS resolution ───────────────────────────────────────
    dns_result = resolve_hostname(hostname)

    # ── Stage 4: SSL/TLS analysis ─────────────────────────────────────
    ssl_result = analyse_ssl_for_url(normalized_url, timeout=settings.connect_timeout)

    # ── Stage 5: Reachability + redirect + HTML ───────────────────────
    reach_result = await fetch_url(normalized_url, follow_redirects=True)
    website_status = reach_result["status"]

    redirect_analysis: dict = {}
    html_analysis: dict = {}
    login_page_analysis: dict = {}

    if website_status not in ("ssrf_blocked", "analysis_error"):
        # Redirect chain analysis.
        redirect_analysis = analyse_redirects(
            initial_url=normalized_url,
            final_url=reach_result.get("final_url", normalized_url),
            redirect_chain=reach_result.get("redirect_chain", []),
        )

        # HTML analysis (only if we got HTML back).
        if reach_result.get("is_html") and reach_result.get("html"):
            html_analysis = analyse_website_html(
                html=reach_result["html"],
                page_url=reach_result.get("final_url", normalized_url),
            )

            # ── Stage 5b: Fake login page detection ──────────────────
            url_features_for_login = extract_url_features(normalized_url)
            login_page_analysis = analyse_login_page(
                html_analysis=html_analysis,
                domain_analysis=domain_analysis,
                spoofing_analysis=spoofing_analysis,
                url_features=url_features_for_login,
            )

    # ── Stage 6: XGBoost ML probability ──────────────────────────────
    ml_probability = 0.0
    model_version = "not_loaded"
    model = registry.get(settings.default_model)

    if model is not None:
        try:
            url_features = extract_url_features(normalized_url)
            ml_probability = model.predict_probability(url_features)
            model_version = f"{model.name}-{model.version}"
        except Exception as exc:
            logger.warning("XGBoost inference error: %s", exc)

    # ── Stage 7: Evidence collection ─────────────────────────────────
    evidence_items: list[EvidenceItem] = collect_evidence(
        url_manipulation=manipulation_signals,
        domain_signals=domain_analysis.get("signals", []),
        dns_signals=dns_result.get("signals", []),
        ssl_signals=ssl_result.get("signals", []),
        redirect_signals=redirect_analysis.get("signals", []),
        html_signals=html_analysis.get("signals", []),
        spoofing_signals=spoofing_analysis.get("signals", []),
        login_page_signals=login_page_analysis.get("signals", []),
        reachability_signals=reach_result.get("signals", []),
    )

    # ── Stage 7b: Threat intelligence ────────────────────────────────
    ti_url_result = await threat_intel.lookup_url(normalized_url)
    ti_domain_result = await threat_intel.lookup_domain(
        domain_analysis.get("registered_domain", hostname)
    )

    # Add threat intel signals if malicious/suspicious.
    if ti_url_result.get("status") in ("malicious", "suspicious"):
        evidence_items.append(EvidenceItem(
            id="threat_intel_url_flagged",
            severity="high" if ti_url_result["status"] == "malicious" else "medium",
            title=f"URL flagged by threat intelligence ({ti_url_result.get('source')})",
            details=ti_url_result.get("detail", ""),
            source="threat_intel",
        ))

    if ti_domain_result.get("status") in ("malicious", "suspicious"):
        evidence_items.append(EvidenceItem(
            id="threat_intel_domain_flagged",
            severity="high" if ti_domain_result["status"] == "malicious" else "medium",
            title=f"Domain flagged by threat intelligence ({ti_domain_result.get('source')})",
            details=ti_domain_result.get("detail", ""),
            source="threat_intel",
        ))

    # ── Stage 8: Risk engine ──────────────────────────────────────────
    risk_result = calculate_risk_v3(
        ml_probability=ml_probability,
        evidence_items=evidence_items,
        website_status=website_status,
        url_valid=validation.valid,
    )

    verdict = risk_result["verdict"]
    risk_score = risk_result["risk_score"]

    # ── Stage 9: Deterministic explanation ───────────────────────────
    det_explanation = build_explanation(
        verdict=verdict,
        risk_score=risk_score,
        ml_probability=ml_probability,
        evidence_items=evidence_items,
        website_status=website_status,
        normalized_url=normalized_url,
    )

    # ── Stage 9b: Optional LLM enhancement ───────────────────────────
    from llm.provider import llm_provider
    final_explanation = await llm_provider.enhance_explanation(
        verdict=verdict,
        risk_score=risk_score,
        ml_probability=ml_probability,
        evidence_items=evidence_items,
        deterministic_explanation=det_explanation,
        url=normalized_url,
    )

    # ── Legacy detections for dashboard compatibility ─────────────────
    legacy_detections = detect_url_rules(
        extract_url_features(normalized_url)
    )

    # ── Build response ────────────────────────────────────────────────
    return {
        "url": raw_url,
        "normalized_url": normalized_url,

        "website_status": website_status,

        "verdict": verdict,
        "risk_score": risk_score,
        "severity": risk_result.get("severity"),

        "ml_probability": round(ml_probability, 6),
        "model_version": model_version,

        "detections": legacy_detections,

        "evidence": [item.as_dict() for item in evidence_items],

        "domain": {
            k: v for k, v in domain_analysis.items() if k != "signals"
        },

        "dns": {
            k: v for k, v in dns_result.items() if k != "signals"
        },

        "ssl": {
            k: v for k, v in ssl_result.items() if k != "signals"
        },

        "redirects": redirect_analysis,

        "website": {
            k: v for k, v in html_analysis.items() if k != "signals"
        },

        "threat_intel": {
            "url": ti_url_result,
            "domain": ti_domain_result,
        },

        "login_page": login_page_analysis,

        "explanation": final_explanation,

        "recommendation": final_explanation.get("recommendation", []),

        "score_breakdown": risk_result.get("score_breakdown", {}),
    }
