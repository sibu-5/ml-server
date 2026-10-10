"""
LLM explanation provider — V3 architecture.

The LLM is an OPTIONAL explanation layer ONLY.

Critical rules
--------------
- The LLM MUST NOT decide the security verdict.
- The LLM MUST NOT invent evidence or visual artifacts.
- The LLM MUST NOT override the deterministic risk engine or ML model output.
- The LLM only converts already-verified evidence/predictions into plain prose.
- Global master switch `settings.llm_enabled` (bool) controls LLM generation.
- When `llm_enabled` is False, zero calls are made to Ollama.
- If Ollama is unavailable, the ML prediction is preserved intact.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

from app.config import settings
from app.evidence import EvidenceItem

logger = logging.getLogger(__name__)


# ── Abstract interface ───────────────────────────────────────────────────────

class LLMExplanationProvider(ABC):
    """
    Interface for LLM explanation providers.
    Provides methods for both Phishing and Deepfake explanation enhancement.
    """

    @abstractmethod
    async def enhance_explanation(
        self,
        *,
        verdict: str,
        risk_score: float,
        ml_probability: float,
        evidence_items: list[EvidenceItem],
        deterministic_explanation: dict,
        url: str,
    ) -> dict:
        """Enhance phishing analysis explanation."""
        pass

    @abstractmethod
    async def enhance_deepfake_explanation(
        self,
        *,
        prediction: str,
        confidence: float,
        risk_level: str,
        details: dict,
        filename: Optional[str] = None,
    ) -> dict:
        """Enhance deepfake image analysis explanation."""
        pass


# ── Qwen3 local provider ─────────────────────────────────────────────────────

class Qwen3LocalProvider(LLMExplanationProvider):
    """
    Shared Qwen3 4B provider via Ollama API (`POST http://{LLM_HOST}:{LLM_PORT}/api/generate`).
    Used for both Phishing and Deepfake explanation enhancement.
    """

    def __init__(self):
        self._base = f"http://{settings.llm_host}:{settings.llm_port}"
        self._model = settings.llm_model
        self._timeout = settings.llm_timeout

    def _build_phishing_prompt(
        self,
        verdict: str,
        risk_score: float,
        ml_probability: float,
        evidence_items: list[EvidenceItem],
        url: str,
    ) -> str:
        evidence_lines = "\n".join(
            f"- [{e.severity.upper()}] {e.title}: {e.details}"
            for e in evidence_items
            if e.severity != "info"
        ) or "No significant evidence signals."

        return f"""You are a cybersecurity assistant explaining a URL analysis result to a non-technical user.

URL: {url}
Verdict: {verdict}
Risk Score: {risk_score:.0f}/100
ML Phishing Probability: {ml_probability:.1%}

Evidence signals detected:
{evidence_lines}

Instructions:
- Write a clear, plain-English explanation of why this URL received the verdict above.
- Do NOT change the verdict. Do NOT invent new evidence.
- Do NOT make up DNS, SSL, or website analysis that was not reported above.
- Be factual, concise, and helpful to a non-technical reader.
- Write 2-3 sentences for the summary.
- Do not use markdown headers.
- Do not add any JSON, code blocks, or structured output.

Write only the explanation paragraph:"""

    def _build_deepfake_prompt(
        self,
        prediction: str,
        confidence: float,
        risk_level: str,
        details: dict,
        filename: Optional[str],
    ) -> str:
        fname = filename or "uploaded image"
        spec_var = details.get("spectral_variance", "N/A")
        hfreq_ratio = details.get("high_freq_ratio", "N/A")
        spatial_noise = details.get("spatial_noise_level", "N/A")

        return f"""You are an AI forensics assistant explaining an image deepfake detection result to a user.

Image: {fname}
Prediction Label: {prediction.upper()}
Confidence Score: {confidence * 100:.1f}%
Assessed Risk Level: {risk_level.upper()}

Technical Spectral & Spatial Findings:
- Spectral Frequency Variance: {spec_var}
- High-Frequency Energy Ratio: {hfreq_ratio}
- Spatial Noise Residual Level: {spatial_noise}

Instructions:
- Write a 2-sentence plain-English explanation of this forensic result.
- State whether synthetic manipulation or frequency anomalies were detected.
- Do NOT change the prediction or state forensic certainty without supporting evidence.
- Do NOT invent fake visual features, specific facial edits, or fake evidence.
- Be professional, factual, and concise.

Write only the explanation paragraph:"""

    async def enhance_explanation(
        self,
        *,
        verdict: str,
        risk_score: float,
        ml_probability: float,
        evidence_items: list[EvidenceItem],
        deterministic_explanation: dict,
        url: str,
    ) -> dict:
        # Strict dynamic check of global LLM master switch
        if not bool(settings.llm_enabled):
            return deterministic_explanation

        try:
            import httpx
            prompt = self._build_phishing_prompt(
                verdict, risk_score, ml_probability, evidence_items, url
            )

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": prompt,
                        "stream": False,
                    },
                )

            if resp.status_code != 200:
                logger.warning(
                    "Ollama returned HTTP %d — returning deterministic explanation.",
                    resp.status_code,
                )
                return deterministic_explanation

            data = resp.json()
            llm_text = (data.get("response") or "").strip()

            if not llm_text:
                return deterministic_explanation

            enhanced = dict(deterministic_explanation)
            enhanced["summary"] = llm_text
            enhanced["llm_enhanced"] = True
            return enhanced

        except Exception as exc:
            logger.warning(
                "Phishing LLM explanation unavailable (%s) — using deterministic explanation.",
                exc,
            )
            return deterministic_explanation

    async def enhance_deepfake_explanation(
        self,
        *,
        prediction: str,
        confidence: float,
        risk_level: str,
        details: dict,
        filename: Optional[str] = None,
    ) -> dict:
        # Strict dynamic check of global LLM master switch
        if not bool(settings.llm_enabled):
            return {
                "summary": f"Image analysis concluded prediction '{prediction}' with confidence {confidence * 100:.1f}%. LLM explanation layer is disabled.",
                "status": "disabled",
            }

        try:
            import httpx
            prompt = self._build_deepfake_prompt(
                prediction, confidence, risk_level, details, filename
            )

            async with httpx.AsyncClient(timeout=settings.deepfake_llm_timeout) as client:
                resp = await client.post(
                    f"{self._base}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": prompt,
                        "stream": False,
                    },
                )

            if resp.status_code != 200:
                logger.warning("Ollama returned HTTP %d for deepfake explanation.", resp.status_code)
                return {
                    "summary": f"Image analysis concluded prediction '{prediction}' with confidence {confidence * 100:.1f}%. LLM explanation unavailable.",
                    "status": "unavailable",
                }

            data = resp.json()
            llm_text = (data.get("response") or "").strip()

            if not llm_text:
                return {
                    "summary": f"Image analysis concluded prediction '{prediction}' with confidence {confidence * 100:.1f}%.",
                    "status": "unavailable",
                }

            return {
                "summary": llm_text,
                "status": "enabled",
            }

        except Exception as exc:
            logger.warning("Deepfake LLM explanation failed (%s) — returning status unavailable.", exc)
            return {
                "summary": f"Image analysis concluded prediction '{prediction}' with confidence {confidence * 100:.1f}%. LLM explanation unavailable.",
                "status": "unavailable",
            }


# Singleton instance
llm_provider = Qwen3LocalProvider()
