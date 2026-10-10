"""
CyberGuard ML Server — Training/Data Pipeline Package.

Auto-registers all feature pipelines on import.
Use PipelineOrchestrator to run any registered pipeline.

Quick start:
    from pipeline import PipelineOrchestrator

    orchestrator = PipelineOrchestrator()
    job = orchestrator.run(
        feature="deepfake",
        modality="image",
        dataset_id="deepfake_image_001",
        model_version="v5",
    )

Model approval:
    from pipeline import approve_model
    approve_model("deepfake", "image", "v5", promote_to_production=True)
"""

# Register all pipelines (deepfake image, future features, etc.)
import pipeline.deepfake.register  # noqa: F401 — side effect: registers pipelines

from pipeline.orchestrator import (   # noqa: F401
    PipelineOrchestrator,
    approve_model,
    reject_model,
    rollback_model,
)

__all__ = [
    "PipelineOrchestrator",
    "approve_model",
    "reject_model",
    "rollback_model",
]
