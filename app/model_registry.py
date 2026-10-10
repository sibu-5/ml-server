"""
CyberGuard ML Server — Central Model Registry.

Loads ALL ML models (phishing, deepfake, and future features) at server startup.
Models are stored in a dict keyed by their registry name and reused for all requests.

Registry JSON schema (models/registry.json)
--------------------------------------------
{
    "models": [
        {
            "name"       : "phishing",           # unique identifier (used in API)
            "feature"    : "phishing",           # feature group
            "version"    : "v3",
            "type"       : "xgboost",            # framework / type
            "media_type" : "url",                # input media type
            "artifact"   : "models/ml-phishing/model.joblib",
            "enabled"    : true
        },
        {
            "name"       : "deepfake_image",
            "feature"    : "deepfake",
            "version"    : "v1",
            "type"       : "pytorch",
            "media_type" : "image",
            "artifact"   : "models/deepfake/image/v1/model.pt",
            "enabled"    : true
        }
    ]
}

Adding a new model
------------------
1. Add an entry to models/registry.json.
2. Restart the server.
3. No code changes required (for known frameworks).

Adding a new framework
-----------------------
1. Create a loader class in app/deepfake/models/base.py.
2. Register the framework string in create_deepfake_model().
3. Add the registry entry.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.models.phishing import PhishingModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central model registry for all ML features.

    Models are loaded once at server startup and reused for requests.
    Future features should be registered here without changing the API layer.

    Lookup key = model "name" field in registry.json.
    """

    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self.models: dict[str, Any] = {}
        self._raw_config: list[dict] = []

    # ── Load ─────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load and initialise all enabled models from the registry JSON."""
        self.models = {}

        if not self.registry_path.exists():
            logger.warning(
                "Model registry not found at %s. No models loaded.",
                self.registry_path,
            )
            return

        try:
            config = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read model registry: %s", exc)
            return

        self._raw_config = config.get("models", [])

        for item in self._raw_config:
            if not item.get("enabled", True):
                continue
            self._load_model(item)

        logger.info(
            "Model registry loaded. Registered: %s",
            list(self.models.keys()),
        )

    def _load_model(self, item: dict) -> None:
        name = item.get("name", "")
        feature = item.get("feature", item.get("type", ""))  # backward-compat

        if not name:
            logger.warning("Registry entry missing 'name' field: %s", item)
            return

        try:
            if feature == "phishing" or name == "phishing":
                self._load_phishing(item)
            elif feature == "deepfake":
                self._load_deepfake(item)
            else:
                # Unknown feature — log and skip.
                # Future: add a plugin registration system here.
                logger.warning(
                    "Unknown feature '%s' for model '%s'. "
                    "Register a loader in ModelRegistry._load_model().",
                    feature, name,
                )
        except Exception as exc:
            logger.error("Failed to load model '%s': %s", name, exc)

    def _load_phishing(self, item: dict) -> None:
        name = item["name"]
        model = PhishingModel(item["artifact"])
        if model.load():
            self.models[name] = model
            logger.info("Phishing model '%s' loaded.", name)
        else:
            logger.warning(
                "Phishing model '%s' artifact not found at %s.",
                name, item.get("artifact"),
            )

    def _load_deepfake(self, item: dict) -> None:
        from app.deepfake.models.base import create_deepfake_model, DeepfakeStubModel

        name = item["name"]
        model = create_deepfake_model(item)

        if model.load():
            self.models[name] = model
            logger.info(
                "Deepfake model '%s' v%s loaded (%s).",
                name, model.version, model.framework,
            )
        else:
            logger.warning(
                "Deepfake model '%s' failed to load. Falling back to stub mode.",
                name,
            )
            stub = DeepfakeStubModel(
                name=name,
                version=item.get("version", "v1"),
                artifact_path=item.get("artifact", ""),
                media_type=item.get("media_type", "unknown"),
                framework="stub",
            )
            stub.load()
            self.models[name] = stub

    # ── Lookup ───────────────────────────────────────────────────────────────

    def get(self, name: str) -> Any:
        """Return the loaded model for *name*, or None if not registered."""
        return self.models.get(name)

    def names(self) -> list[str]:
        """Return list of all registered model names."""
        return list(self.models.keys())

    def names_by_feature(self, feature: str) -> list[str]:
        """Return registered model names for a specific feature (e.g. 'deepfake')."""
        result = []
        for item in self._raw_config:
            if item.get("feature") == feature and item.get("name") in self.models:
                result.append(item["name"])
        return result

    def get_registry_metadata(self) -> list[dict]:
        """Return the raw registry config (for /health or admin endpoints)."""
        return [
            {
                "name": item.get("name"),
                "feature": item.get("feature"),
                "version": item.get("version"),
                "framework": item.get("type"),
                "media_type": item.get("media_type"),
                "enabled": item.get("enabled", True),
                "loaded": item.get("name") in self.models,
            }
            for item in self._raw_config
        ]
