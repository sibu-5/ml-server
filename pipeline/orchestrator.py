"""
CyberGuard ML Server — Pipeline Orchestrator.

The orchestrator is the single entry point for running a complete ML pipeline.
It routes datasets to the correct feature pipeline, runs all steps in order,
tracks job state, and records the result.

Usage
-----
    from pipeline.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator()
    job = orchestrator.run(
        feature="deepfake",
        modality="image",
        dataset_id="deepfake_image_001",
        dataset_version="v1",
        model_version="v5",
    )

The orchestrator does NOT:
- Modify raw data
- Overwrite production models
- Auto-promote candidates without human approval

It produces a report and leaves the model in "candidate" status.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.base import (
    DatasetManifest,
    JobState,
    ModelStatus,
    PipelineRegistry,
    ValidationResult,
    PreprocessingResult,
    SplitResult,
    TrainingResult,
    EvaluationResult,
    pipeline_registry,
)
from pipeline.job_tracker import Job, create_job

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


class PipelineOrchestrator:
    """
    Feature-aware ML pipeline orchestrator.

    Resolves (feature, modality) → PipelineBundle, then runs all
    pipeline steps in sequence while tracking job state.

    Error handling
    -------------
    - Any step failure marks the job FAILED and preserves all previous outputs.
    - Production model is never touched.
    - A detailed job log is written throughout.
    """

    def __init__(self, registry: PipelineRegistry | None = None):
        self._registry = registry or pipeline_registry

    def run(
        self,
        feature: str,
        modality: str,
        dataset_id: str,
        dataset_version: str = "v1",
        model_version: str = "v5",
        config: dict[str, Any] | None = None,
        manifest_path: Path | None = None,
    ) -> Job:
        """
        Run the full pipeline for a given feature + modality.

        Parameters
        ----------
        feature          : e.g. "deepfake", "phishing"
        modality         : e.g. "image", "url", "audio"
        dataset_id       : Unique dataset identifier
        dataset_version  : Version tag for this dataset
        model_version    : Target version for new candidate model
        config           : Override training hyperparameters
        manifest_path    : Optional path to dataset_manifest.yaml/.json

        Returns
        -------
        Job object (persisted to disk).
        """
        config = config or {}

        # Create tracking job
        job = create_job(
            feature=feature,
            modality=modality,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            model_version=model_version,
        )
        job.log(f"Pipeline started: feature={feature} modality={modality} "
                f"dataset={dataset_id} target_version={model_version}")

        # Resolve pipeline bundle
        bundle = self._registry.get(feature, modality)
        if bundle is None:
            job.fail(
                f"No pipeline registered for feature='{feature}' modality='{modality}'. "
                f"Available: {self._registry.list_pipelines()}"
            )
            return job

        job.log(f"Pipeline bundle resolved: {bundle.feature}.{bundle.modality}")

        # Build manifest
        manifest = self._load_or_build_manifest(
            feature, modality, dataset_id, dataset_version,
            manifest_path=manifest_path,
        )
        job.metadata["manifest"] = {
            "dataset_id": manifest.dataset_id,
            "feature": manifest.feature,
            "modality": manifest.modality,
            "pipeline_key": manifest.pipeline_key(),
        }
        job.save()

        # Resolve data paths
        raw_root       = ROOT / "data" / "raw" / feature / modality
        processed_root = ROOT / "data" / "processed" / feature / modality
        trained_root   = ROOT / "data" / "trained" / feature / modality / dataset_version
        report_root    = ROOT / "reports" / feature / modality / "training"
        model_root     = ROOT / "models" / feature / modality / model_version

        processed_root.mkdir(parents=True, exist_ok=True)
        trained_root.mkdir(parents=True, exist_ok=True)
        report_root.mkdir(parents=True, exist_ok=True)
        model_root.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Validate ───────────────────────────────────────────────
        job.transition(JobState.VALIDATING, note="Running dataset validation")
        try:
            validator = bundle.validator()
            val_result = validator.validate(manifest, raw_root)
        except Exception as exc:
            job.fail(f"Validator crashed: {exc}")
            logger.exception("Validator crashed for job '%s'.", job.job_id)
            return job

        self._save_step_result(job, "validation", val_result)

        if not val_result.passed:
            job.log("Validation FAILED. Saving report and stopping.", "ERROR")
            job.log(f"Errors: {val_result.errors}", "ERROR")
            self._save_validation_failure_report(
                manifest, val_result,
                ROOT / "reports" / feature / modality / "datasets",
            )
            job.fail(f"Dataset validation failed: {'; '.join(val_result.errors)}")
            return job

        job.log(f"Validation passed. Stats: {val_result.stats}")

        # ── Step 2: Preprocess ─────────────────────────────────────────────
        job.transition(JobState.PREPROCESSING, note="Preprocessing dataset")
        try:
            preprocessor = bundle.preprocessor()
            prep_result = preprocessor.preprocess(manifest, raw_root, processed_root)
        except Exception as exc:
            job.fail(f"Preprocessor crashed: {exc}")
            logger.exception("Preprocessor crashed for job '%s'.", job.job_id)
            return job

        self._save_step_result(job, "preprocessing", prep_result)

        if not prep_result.success:
            job.fail(f"Preprocessing failed: {'; '.join(prep_result.errors)}")
            return job

        job.log(f"Preprocessing complete. Stats: {prep_result.stats}")

        # ── Step 3: Split ──────────────────────────────────────────────────
        job.transition(JobState.SPLITTING, note="Creating train/val/test split")
        try:
            splitter = bundle.splitter()
            split_result = splitter.split(manifest, processed_root, trained_root)
        except Exception as exc:
            job.fail(f"Splitter crashed: {exc}")
            logger.exception("Splitter crashed for job '%s'.", job.job_id)
            return job

        self._save_step_result(job, "split", split_result)

        if not split_result.success:
            job.fail(f"Dataset split failed: {'; '.join(split_result.errors)}")
            return job

        job.log(f"Split complete. Stats: {split_result.stats}")

        # ── Step 4: Train ──────────────────────────────────────────────────
        job.transition(JobState.TRAINING, note="Training candidate model")
        try:
            trainer = bundle.trainer()
            train_result = trainer.train(manifest, split_result, model_version, config)
        except Exception as exc:
            job.fail(f"Trainer crashed: {exc}")
            logger.exception("Trainer crashed for job '%s'.", job.job_id)
            return job

        self._save_step_result(job, "training", train_result)

        if not train_result.success:
            job.fail(f"Training failed: {'; '.join(train_result.errors)}")
            return job

        job.log(f"Training complete. Artifact: {train_result.artifact_path}")

        # ── Step 5: Evaluate ───────────────────────────────────────────────
        job.transition(JobState.EVALUATING, note="Evaluating candidate model")
        try:
            evaluator = bundle.evaluator()
            eval_result = evaluator.evaluate(
                manifest,
                train_result.artifact_path,
                split_result.test_dir,
                model_version,
            )
        except Exception as exc:
            job.fail(f"Evaluator crashed: {exc}")
            logger.exception("Evaluator crashed for job '%s'.", job.job_id)
            return job

        self._save_step_result(job, "evaluation", eval_result)

        if not eval_result.success:
            job.log(f"Evaluation issues: {eval_result.errors}", "WARNING")

        job.log(f"Evaluation complete. Metrics: {eval_result.metrics}")

        # ── Step 6: Previous production metrics for comparison ────────────
        prev_metrics = self._load_production_metrics(feature, modality)

        # ── Step 7: Generate report ────────────────────────────────────────
        job.transition(JobState.GENERATING_REPORT)
        try:
            report_gen = bundle.report_gen()
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            run_report_dir = report_root / f"{feature}_{modality}_{model_version}_{ts}"
            report_path = report_gen.generate(
                manifest,
                val_result,
                prep_result,
                split_result,
                train_result,
                eval_result,
                model_version,
                run_report_dir,
                previous_metrics=prev_metrics,
            )
            job.metadata["report_path"] = str(report_path)
            job.save()
            job.log(f"Report generated at: {report_path}")
        except Exception as exc:
            job.log(f"Report generation failed (non-fatal): {exc}", "WARNING")
            logger.warning("Report generation failed: %s", exc)

        # ── Step 8: Register candidate in model registry ──────────────────
        if train_result.artifact_path and train_result.artifact_path.exists():
            self._register_candidate(
                feature, modality, model_version,
                train_result, eval_result, manifest,
                job.metadata.get("report_path", ""),
            )
            job.log(f"Candidate model registered: {feature}_{modality} v{model_version}")
        else:
            job.log("Artifact not found — skipping registry update.", "WARNING")

        # ── Step 9: Wait for human approval ───────────────────────────────
        job.complete(
            JobState.WAITING_FOR_APPROVAL,
            note=(
                f"Candidate model {feature}_{modality} v{model_version} ready for review. "
                "Use approve_model() or reject_model() to proceed."
            ),
        )
        return job

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_or_build_manifest(
        feature: str,
        modality: str,
        dataset_id: str,
        dataset_version: str,
        manifest_path: Path | None = None,
    ) -> DatasetManifest:
        """Load manifest from file or build a minimal one from args."""
        if manifest_path and manifest_path.exists():
            try:
                import yaml  # type: ignore[import]
                data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                return DatasetManifest(**data)
            except Exception as exc:
                logger.warning("Failed to load manifest from %s: %s — using defaults.", manifest_path, exc)

        # Build sensible default
        default_labels = {
            "image": ["real", "fake"],
            "video": ["real", "fake"],
            "audio": ["real", "fake"],
            "url":   ["phishing", "legitimate"],
            "email": ["phishing", "legitimate"],
        }
        return DatasetManifest(
            dataset_id=dataset_id,
            feature=feature,
            modality=modality,
            task="binary_classification",
            labels=default_labels.get(modality, ["class_0", "class_1"]),
            version=dataset_version,
            source="local",
        )

    @staticmethod
    def _save_step_result(job: Job, step_name: str, result: Any) -> None:
        """Serialise a step result dataclass and attach to job metadata."""
        try:
            from dataclasses import asdict
            job.metadata[step_name] = asdict(result)
            job.save()
        except Exception:
            pass

    @staticmethod
    def _save_validation_failure_report(
        manifest: DatasetManifest,
        val_result: ValidationResult,
        report_dir: Path,
    ) -> None:
        """Save a failure report when validation does not pass."""
        report_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = report_dir / f"validation_failure_{manifest.dataset_id}_{ts}.json"
        content = {
            "dataset_id": manifest.dataset_id,
            "feature": manifest.feature,
            "modality": manifest.modality,
            "timestamp": ts,
            "passed": False,
            "errors": val_result.errors,
            "warnings": val_result.warnings,
            "stats": val_result.stats,
            "recommendation": "FAILED_VALIDATION",
        }
        path.write_text(json.dumps(content, indent=2), encoding="utf-8")
        logger.info("Validation failure report saved to %s", path)

    @staticmethod
    def _load_production_metrics(feature: str, modality: str) -> dict[str, Any] | None:
        """Load metrics from the current production model if available."""
        candidate_registry = ROOT / "models" / feature / modality / "model_versions.json"
        if not candidate_registry.exists():
            return None
        try:
            data = json.loads(candidate_registry.read_text(encoding="utf-8"))
            for entry in data.get("versions", []):
                if entry.get("status") == ModelStatus.PRODUCTION.value:
                    return entry.get("metrics", {})
        except Exception:
            pass
        return None

    @staticmethod
    def _register_candidate(
        feature: str,
        modality: str,
        model_version: str,
        train_result: TrainingResult,
        eval_result: EvaluationResult,
        manifest: DatasetManifest,
        report_path: str,
    ) -> None:
        """Register the candidate model in the per-feature version registry."""
        from datetime import datetime, timezone

        versions_path = ROOT / "models" / feature / modality / "model_versions.json"
        versions_path.parent.mkdir(parents=True, exist_ok=True)

        if versions_path.exists():
            data = json.loads(versions_path.read_text(encoding="utf-8"))
        else:
            data = {"feature": feature, "modality": modality, "versions": []}

        entry = {
            "model_id": f"{feature}_{modality}",
            "version": model_version,
            "feature": feature,
            "modality": modality,
            "framework": "pytorch",
            "artifact_path": str(train_result.artifact_path),
            "dataset_id": manifest.dataset_id,
            "dataset_version": manifest.version,
            "metrics": eval_result.metrics,
            "status": ModelStatus.CANDIDATE.value,
            "report_path": report_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "approved_at": None,
        }

        # Replace if same version already exists
        data["versions"] = [
            v for v in data["versions"] if v.get("version") != model_version
        ]
        data["versions"].append(entry)
        versions_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Model approval / promotion ─────────────────────────────────────────────────

def approve_model(
    feature: str,
    modality: str,
    version: str,
    promote_to_production: bool = False,
) -> dict[str, Any]:
    """
    Approve a candidate model and optionally promote it to production.

    When promoted:
    - Previous production model → archived
    - Candidate → production
    - models/registry.json updated
    - Rollback remains possible (archived model still on disk)

    Parameters
    ----------
    promote_to_production : If True, also promote to production immediately.
                            Default False (human can approve first, promote later).
    """
    versions_path = ROOT / "models" / feature / modality / "model_versions.json"
    if not versions_path.exists():
        return {"error": f"No model version registry found at {versions_path}"}

    data = json.loads(versions_path.read_text(encoding="utf-8"))
    versions = data.get("versions", [])

    target = next((v for v in versions if v["version"] == version), None)
    if target is None:
        return {"error": f"Version '{version}' not found for {feature}/{modality}"}

    now = datetime.now(timezone.utc).isoformat()
    target["approved_at"] = now

    if promote_to_production:
        # Archive current production
        for v in versions:
            if v.get("status") == ModelStatus.PRODUCTION.value:
                v["status"] = ModelStatus.ARCHIVED.value
                logger.info("Archived previous production: %s v%s", feature, v["version"])

        target["status"] = ModelStatus.PRODUCTION.value
        target["promoted_at"] = now

        # Update inference registry
        _update_inference_registry(feature, modality, version, target["artifact_path"])
        logger.info("Model promoted to production: %s/%s v%s", feature, modality, version)
    else:
        target["status"] = ModelStatus.APPROVED.value

    data["versions"] = versions
    versions_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return {
        "success": True,
        "model_id": f"{feature}_{modality}",
        "version": version,
        "status": target["status"],
        "promoted": promote_to_production,
    }


def reject_model(
    feature: str,
    modality: str,
    version: str,
    reason: str = "",
) -> dict[str, Any]:
    """Reject a candidate model."""
    versions_path = ROOT / "models" / feature / modality / "model_versions.json"
    if not versions_path.exists():
        return {"error": f"No model version registry at {versions_path}"}

    data = json.loads(versions_path.read_text(encoding="utf-8"))
    versions = data.get("versions", [])

    target = next((v for v in versions if v["version"] == version), None)
    if target is None:
        return {"error": f"Version '{version}' not found"}

    target["status"] = ModelStatus.REJECTED.value
    target["rejection_reason"] = reason
    target["rejected_at"] = datetime.now(timezone.utc).isoformat()

    versions_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"success": True, "version": version, "status": "rejected"}


def rollback_model(feature: str, modality: str, to_version: str) -> dict[str, Any]:
    """
    Roll back production to a previously archived model version.

    The current production model is archived, and the target version
    is set to production. The model artifact is NOT deleted.
    """
    return approve_model(feature, modality, to_version, promote_to_production=True)


def _update_inference_registry(
    feature: str,
    modality: str,
    version: str,
    artifact_path: str,
) -> None:
    """Update models/registry.json to point to the new production model."""
    registry_path = ROOT / "models" / "registry.json"
    if not registry_path.exists():
        logger.warning("models/registry.json not found — skipping inference registry update.")
        return

    config = json.loads(registry_path.read_text(encoding="utf-8"))
    model_name = f"{feature}_{modality}"

    p = Path(artifact_path)
    if p.is_absolute():
        try:
            rel_artifact = str(p.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            rel_artifact = str(p).replace("\\", "/")
    else:
        rel_artifact = str(p).replace("\\", "/")

    updated = False
    for item in config.get("models", []):
        if item.get("name") == model_name:
            item["version"] = version
            item["artifact"] = rel_artifact
            item["enabled"] = True
            updated = True
            break

    if not updated:
        config.setdefault("models", []).append({
            "name": model_name,
            "feature": feature,
            "version": version,
            "type": "pytorch",
            "media_type": modality,
            "artifact": rel_artifact,
            "enabled": True,
        })

    registry_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    logger.info("Inference registry updated: %s → v%s", model_name, version)
