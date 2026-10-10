"""
ThreatForge ML Server — V3 Configuration.

All runtime values are read from environment variables / .env.
Never hard-code secrets, keys, ports, or URLs here.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Core server ────────────────────────────────────────────────
    ml_api_key: str = ""
    host: str = "0.0.0.0"
    port: int = 8001

    # ── Model registry ────────────────────────────────────────────
    model_registry_file: str = "models/registry.json"
    default_model: str = "phishing"

    # ── URL / request limits ──────────────────────────────────────
    max_url_length: int = 4096
    connect_timeout: float = 5.0       # seconds
    read_timeout: float = 10.0         # seconds
    request_timeout: float = 20.0      # total wall-clock limit
    max_redirects: int = 5
    max_response_bytes: int = 5_242_880   # 5 MB
    max_html_bytes: int = 2_097_152       # 2 MB

    # ── Threat intelligence ───────────────────────────────────────
    # Set to a real API key to enable VirusTotal lookups.
    virustotal_api_key: str = ""
    # Generic threat-intel provider name (e.g. "virustotal", "none").
    threat_intel_provider: str = "none"

    # ── LLM explanation layer ─────────────────────────────────────
    # Local Qwen3 / other LLM endpoint (future).
    llm_enabled: bool = False
    llm_host: str = "127.0.0.1"
    llm_port: int = 11434
    llm_model: str = "qwen3:4b"
    llm_timeout: float = 30.0

    # ── CORS (for frontend browser uploads) ──────────────────────────
    # Comma-separated list of allowed origins, or "*" for any origin.
    # Example: "http://localhost:3000,https://app.threatforge.io"
    cors_origins: str = "*"

    # ── Deepfake image detection settings ─────────────────────────────
    deepfake_max_image_bytes: int = 20 * 1024 * 1024       # 20 MB max upload
    deepfake_max_image_pixels: int = 25_000_000             # 25 MP max (decompression bomb protection)
    deepfake_max_concurrent_inferences: int = 2             # Concurrency limit for inference
    deepfake_llm_explanation_enabled: bool = True          # Enable deepfake explanation when LLM_ENABLED is true
    deepfake_llm_timeout: float = 20.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
