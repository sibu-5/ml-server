import asyncio
import json
import time
import uuid
import logging
from typing import AsyncGenerator

from app.config import settings
from app.features.url import (
    normalize_url,
    validate_url,
    detect_url_manipulation,
    extract_url_features
)
from app.features.domain import analyse_domain
from app.features.dns import resolve_hostname
from app.analyzers.ssl import analyse_ssl_for_url
from app.analyzers.reachability import fetch_url
from app.analyzers.redirect import analyse_redirects
from app.analyzers.website import analyse_website_html
from app.analyzers.login_page import analyse_login_page
from app.analyzers.spoofing import analyse_spoofing
from app.intelligence.threat_intel import threat_intel
from app.rules import detect_url_rules
from app.evidence import collect as collect_evidence, EvidenceItem
from app.risk import calculate_risk_v3
from app.explanation import build_explanation
from llm.provider import llm_provider

logger = logging.getLogger(__name__)

async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    import functools
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

async def analyze_url_stream_generator(raw_url: str, model=None) -> AsyncGenerator[str, None]:
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    queue = asyncio.Queue()

    def send_event(stage: str, status: str, data: dict = None):
        event = {
            "request_id": request_id,
            "stage": stage,
            "status": status,
            "timestamp": time.time(),
        }
        if data is not None:
            event["data"] = data
        payload = f"event: {stage}\ndata: {json.dumps(event)}\n\n"
        queue.put_nowait(payload)
        
    send_event("analysis", "started")
    
    # ── Stage 1: URL ───────────────────────
    send_event("url", "started")
    normalized_url = normalize_url(raw_url)
    validation = validate_url(raw_url, max_length=settings.max_url_length)
    manipulation_signals = detect_url_manipulation(raw_url)
    url_features = extract_url_features(normalized_url)
    
    send_event("url", "completed", {
        "normalized_url": normalized_url,
        "valid": validation.valid,
        "detail": validation.detail
    })

    if not validation.valid:
        send_event("analysis", "completed", {"error": "INVALID_URL", "total_time": time.time() - start_time})
        queue.put_nowait(None)
        while True:
            msg = await queue.get()
            if msg is None:
                break
            yield msg
        return

    hostname = validation.hostname
    results = {}
    
    async def task_domain():
        send_event("domain", "started")
        res = await run_in_thread(analyse_domain, normalized_url)
        results["domain"] = res
        send_event("domain", "completed", {k: v for k, v in res.items() if k != "signals"})
        return res
        
    async def task_dns():
        send_event("dns", "started")
        res = await run_in_thread(resolve_hostname, hostname)
        results["dns"] = res
        send_event("dns", "completed", {k: v for k, v in res.items() if k != "signals"})
        return res

    async def task_ssl():
        send_event("ssl", "started")
        res = await run_in_thread(analyse_ssl_for_url, normalized_url, settings.connect_timeout)
        results["ssl"] = res
        send_event("ssl", "completed", {k: v for k, v in res.items() if k != "signals"})
        return res

    async def task_spoofing():
        send_event("spoofing", "started")
        res = await run_in_thread(analyse_spoofing, normalized_url)
        results["spoofing"] = res
        send_event("spoofing", "completed", {k: v for k, v in res.items() if k != "signals"})
        return res
        
    async def task_threat_intel():
        send_event("threat_intelligence", "started")
        try:
            ti_url = await asyncio.wait_for(threat_intel.lookup_url(normalized_url), timeout=10.0)
            ti_domain = await asyncio.wait_for(threat_intel.lookup_domain(hostname), timeout=10.0)
        except Exception as e:
            ti_url = {"status": "not_configured"}
            ti_domain = {"status": "not_configured"}
            send_event("threat_intelligence", "failed", {"error": str(e)})
            
        res = {"url": ti_url, "domain": ti_domain}
        results["threat_intel"] = res
        send_event("threat_intelligence", "completed", res)
        return res

    async def task_website():
        send_event("website", "started")
        try:
            reach_result = await asyncio.wait_for(
                fetch_url(normalized_url, follow_redirects=True), 
                timeout=settings.request_timeout + 5.0
            )
        except Exception as e:
            reach_result = {"status": "analysis_error", "signals": []}
            send_event("website", "failed", {"error": str(e)})

        website_status = reach_result.get("status", "analysis_error")
        results["reach_result"] = reach_result
        results["website_status"] = website_status
        
        send_event("website", "completed", {
            "website_status": website_status,
            "is_html": reach_result.get("is_html", False)
        })
        
        send_event("redirect", "started")
        if website_status not in ("ssrf_blocked", "analysis_error"):
            redirect_analysis = await run_in_thread(
                analyse_redirects,
                initial_url=normalized_url,
                final_url=reach_result.get("final_url", normalized_url),
                redirect_chain=reach_result.get("redirect_chain", [])
            )
        else:
            redirect_analysis = {}
        results["redirect"] = redirect_analysis
        send_event("redirect", "completed", redirect_analysis)
        return reach_result

    async def task_ml():
        send_event("ml", "started")
        ml_probability = 0.0
        model_version = "not_loaded"
        if model is not None:
            try:
                ml_probability = await run_in_thread(model.predict_probability, url_features)
                model_version = f"{model.name}-{model.version}"
            except Exception as exc:
                logger.warning("XGBoost error: %s", exc)
                send_event("ml", "failed", {"error": str(exc)})
        
        res = {"ml_probability": round(ml_probability, 6), "model_version": model_version}
        results["ml"] = res
        send_event("ml", "completed", res)
        return res

    tasks = [
        asyncio.create_task(task_domain()),
        asyncio.create_task(task_dns()),
        asyncio.create_task(task_ssl()),
        asyncio.create_task(task_spoofing()),
        asyncio.create_task(task_threat_intel()),
        asyncio.create_task(task_website()),
        asyncio.create_task(task_ml()),
    ]
    
    async def orchestrator():
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            
            reach_result = results.get("reach_result", {})
            website_status = results.get("website_status", "analysis_error")
            domain_analysis = results.get("domain", {})
            spoofing_analysis = results.get("spoofing", {})
            
            html_analysis = {}
            login_page_analysis = {}
            
            if website_status not in ("ssrf_blocked", "analysis_error"):
                if reach_result.get("is_html") and reach_result.get("html"):
                    html_analysis = await run_in_thread(
                        analyse_website_html,
                        reach_result["html"],
                        reach_result.get("final_url", normalized_url)
                    )
                    
                    send_event("login_page", "started")
                    login_page_analysis = await run_in_thread(
                        analyse_login_page,
                        html_analysis,
                        domain_analysis,
                        spoofing_analysis,
                        url_features
                    )
                    send_event("login_page", "completed", login_page_analysis)
                    
            results["html"] = html_analysis
            results["login_page"] = login_page_analysis
            
            send_event("evidence", "started")
            evidence_items = await run_in_thread(
                lambda: collect_evidence(
                    url_manipulation=manipulation_signals,
                    domain_signals=domain_analysis.get("signals", []),
                    dns_signals=results.get("dns", {}).get("signals", []),
                    ssl_signals=results.get("ssl", {}).get("signals", []),
                    redirect_signals=results.get("redirect", {}).get("signals", []),
                    html_signals=html_analysis.get("signals", []),
                    spoofing_signals=spoofing_analysis.get("signals", []),
                    login_page_signals=login_page_analysis.get("signals", []),
                    reachability_signals=reach_result.get("signals", [])
                )
            )
            
            ti_url_result = results.get("threat_intel", {}).get("url", {})
            ti_domain_result = results.get("threat_intel", {}).get("domain", {})
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
                
            send_event("evidence", "completed", {"count": len(evidence_items)})
            
            send_event("risk", "started")
            risk_result = await run_in_thread(
                lambda: calculate_risk_v3(
                    ml_probability=results.get("ml", {}).get("ml_probability", 0.0),
                    evidence_items=evidence_items,
                    website_status=website_status,
                    url_valid=validation.valid,
                )
            )
            send_event("risk", "completed", risk_result)
            
            verdict = risk_result["verdict"]
            risk_score = risk_result["risk_score"]
            
            send_event("explanation", "started")
            det_explanation = await run_in_thread(
                lambda: build_explanation(
                    verdict=verdict,
                    risk_score=risk_score,
                    ml_probability=results.get("ml", {}).get("ml_probability", 0.0),
                    evidence_items=evidence_items,
                    website_status=website_status,
                    normalized_url=normalized_url,
                )
            )
            
            try:
                final_explanation = await asyncio.wait_for(
                    llm_provider.enhance_explanation(
                        verdict=verdict,
                        risk_score=risk_score,
                        ml_probability=results.get("ml", {}).get("ml_probability", 0.0),
                        evidence_items=evidence_items,
                        deterministic_explanation=det_explanation,
                        url=normalized_url
                    ),
                    timeout=settings.llm_timeout + 5.0
                )
            except Exception as e:
                send_event("explanation", "failed", {"error": str(e)})
                final_explanation = det_explanation
                
            send_event("explanation", "completed", final_explanation)
            
            final_response = {
                "url": raw_url,
                "normalized_url": normalized_url,
                "website_status": website_status,
                "verdict": verdict,
                "risk_score": risk_score,
                "severity": risk_result.get("severity"),
                "ml_probability": results.get("ml", {}).get("ml_probability", 0.0),
                "model_version": results.get("ml", {}).get("model_version", "not_loaded"),
                "detections": detect_url_rules(url_features),
                "evidence": [item.as_dict() for item in evidence_items],
                "domain": {k: v for k, v in domain_analysis.items() if k != "signals"},
                "dns": {k: v for k, v in results.get("dns", {}).items() if k != "signals"},
                "ssl": {k: v for k, v in results.get("ssl", {}).items() if k != "signals"},
                "redirects": results.get("redirect", {}),
                "website": {k: v for k, v in html_analysis.items() if k != "signals"},
                "threat_intel": results.get("threat_intel", {}),
                "login_page": login_page_analysis,
                "explanation": final_explanation,
                "recommendation": final_explanation.get("recommendation", []),
                "score_breakdown": risk_result.get("score_breakdown", {}),
            }
            send_event("response", "completed", final_response)
            send_event("analysis", "completed", {"total_time": time.time() - start_time})
            
        except Exception as e:
            logger.exception("Error in stream orchestrator")
            send_event("analysis", "failed", {"error": str(e)})
        finally:
            queue.put_nowait(None)
            
    asyncio.create_task(orchestrator())
    
    while True:
        msg = await queue.get()
        if msg is None:
            break
        yield msg
