"""
CyberGuard ML Server — Pipeline Base Classes.

This module defines the abstract interfaces that every ML pipeline
component must implement. Adding a new feature (Deepfake Audio, Malware,
Fraud) requires implementing these interfaces — nothing else changes.

Architecture:

    PipelineComponent (base)
        ├── DatasetValidator
        ├── DatasetPreprocessor
        ├── DatasetSplitter
        ├── ModelTrainer
        ├── ModelEvaluator
        └── ReportGenerator

    PipelineOrchestrator
        └── Uses a registered pipeline (feature + modality) to run all steps.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Job states ────────────────────────────────────────────────────────────────

class JobState(str, Enum):
    QUEUED              = "QUEUED"
    VALIDATING          = "VALIDATING"
    PREPROCESSING       = "PREPROCESSING"
    SPLITTING           = "SPLITTING"
    TRAINING            = "TRAINING"
    EVALUATING          = "EVALUATING"
    GENERATING_REPORT   = "GENERATING_REPORT"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    APPROVED            = "APPROVED"
    REJECTED            = "REJECTED"
    PROMOTED            = "PROMOTED"
    FAILED              = "FAILED"


# ── Model promotion statuses ──────────────────────────────────────────────────

class ModelStatus(str, Enum):
    DEVELOPMENT = "development"
    CANDIDATE   = "candidate"
    TESTING     = "testing"
    APPROVED    = "approved"
    PRODUCTION  = "production"
    REJECTED    = "rejected"
    ARCHIVED    = "archived"


# ── Report recommendations ────────────────────────────────────────────────────

class ReportRecommendation(str, Enum):
    READY_FOR_TESTING    = "READY_FOR_TESTING"
    READY_FOR_PRODUCTION = "READY_FOR_PRODUCTION"
    NEEDS_IMPROVEMENT    = "NEEDS_IMPROVEMENT"
    REJECTED             = "REJECTED"
    FAILED_VALIDATION    = "FAILED_VALIDATION"


# ── Dataset manifest ──────────────────────────────────────────────────────────

@dataclass
class DatasetManifest:
    """
    Describes a dataset and determines which pipeline handles it.

    Every dataset pushed through the pipeline must have a manifest.
    The (feature, modality) pair routes the dataset to the correct pipeline.
    """
    dataset_id: str
    feature: str            # "deepfake", "phishing", "malware", …
    modality: str           # "image", "video", "audio", "url", "email", …
    task: str               # "binary_classification", "multiclass", …
    labels: list[str]       # ["real", "fake"] or ["phishing", "legitimate"]
    version: str            = "v1"
    source: str             = "unknown"
    creation_date: str      = ""
    validation_status: str  = "pending"
    target_model: str       = ""
    extra: dict[str, Any]   = field(default_factory=dict)

    def pipeline_key(self) -> str:
        """Return the routing key: 'deepfake.image', 'phishing.url', etc."""
        return f"{self.feature}.{self.modality}"


# ── Validation result ─────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of dataset validation."""
    passed: bool
    errors: list[str]       = field(default_factory=list)
    warnings: list[str]     = field(default_factory=list)
    stats: dict[str, Any]   = field(default_factory=dict)
    report_path: Path | None = None


# ── Preprocessing result ──────────────────────────────────────────────────────

@dataclass
class PreprocessingResult:
    """Result of dataset preprocessing."""
    success: bool
    output_dir: Path | None = None
    stats: dict[str, Any]   = field(default_factory=dict)
    errors: list[str]       = field(default_factory=list)
    warnings: list[str]     = field(default_factory=list)


# ── Split result ──────────────────────────────────────────────────────────────

@dataclass
class SplitResult:
    """Result of train/val/test split."""
    success: bool
    train_dir: Path | None  = None
    val_dir: Path | None    = None
    test_dir: Path | None   = None
    stats: dict[str, Any]   = field(default_factory=dict)
    errors: list[str]       = field(default_factory=list)


# ── Training result ───────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    """Result of model training."""
    success: bool
    artifact_path: Path | None  = None
    model_version: str          = ""
    training_stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str]           = field(default_factory=list)


# ── Evaluation result ─────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    success: bool
    metrics: dict[str, Any]         = field(default_factory=dict)
    per_class_metrics: dict[str, Any] = field(default_factory=dict)
    confusion_matrix: list | None   = None
    errors: list[str]               = field(default_factory=list)


# ── Abstract pipeline components ──────────────────────────────────────────────

class DatasetValidator(ABC):
    """
    Abstract dataset validator.

    Subclass this for each feature+modality combination:
        DeepfakeImageValidator
        PhishingURLValidator
        MalwareValidator
        …
    """

    @abstractmethod
    def validate(
        self,
        manifest: DatasetManifest,
        raw_data_root: Path,
    ) -> ValidationResult:
        """
        Validate the raw dataset.

        Must NOT modify any data in raw_data_root.
        Returns ValidationResult with passed=False if critical issues found.
        """


class DatasetPreprocessor(ABC):
    """Abstract dataset preprocessor."""

    @abstractmethod
    def preprocess(
        self,
        manifest: DatasetManifest,
        raw_data_root: Path,
        output_dir: Path,
    ) -> PreprocessingResult:
        """
        Preprocess raw data and write artifacts to output_dir.

        raw_data_root is READ-ONLY.
        All processed output goes to output_dir.
        """


class DatasetSplitter(ABC):
    """Abstract train/val/test splitter."""

    @abstractmethod
    def split(
        self,
        manifest: DatasetManifest,
        processed_dir: Path,
        output_dir: Path,
        train_ratio: float = 0.70,
        val_ratio: float   = 0.15,
        test_ratio: float  = 0.15,
        seed: int          = 42,
    ) -> SplitResult:
        """Split processed data into train/val/test directories."""


class ModelTrainer(ABC):
    """Abstract model trainer."""

    @abstractmethod
    def train(
        self,
        manifest: DatasetManifest,
        split_result: SplitResult,
        model_version: str,
        config: dict[str, Any],
    ) -> TrainingResult:
        """Train model from split data and return artifact path."""


class ModelEvaluator(ABC):
    """Abstract model evaluator."""

    @abstractmethod
    def evaluate(
        self,
        manifest: DatasetManifest,
        artifact_path: Path,
        test_dir: Path,
        model_version: str,
    ) -> EvaluationResult:
        """Evaluate model on test set and return metrics."""


class ReportGenerator(ABC):
    """Abstract report generator."""

    @abstractmethod
    def generate(
        self,
        manifest: DatasetManifest,
        validation_result: ValidationResult,
        preprocessing_result: PreprocessingResult,
        split_result: SplitResult,
        training_result: TrainingResult,
        evaluation_result: EvaluationResult,
        model_version: str,
        report_dir: Path,
        previous_metrics: dict[str, Any] | None = None,
    ) -> Path:
        """Generate a complete training report. Returns path to report dir."""


# ── Pipeline registration system ──────────────────────────────────────────────

@dataclass
class PipelineBundle:
    """
    A complete ML pipeline for one (feature, modality) pair.

    Register new pipelines using PipelineRegistry.register().
    The orchestrator looks up the bundle and runs each component in order.
    """
    feature:       str
    modality:      str
    validator:     type[DatasetValidator]
    preprocessor:  type[DatasetPreprocessor]
    splitter:      type[DatasetSplitter]
    trainer:       type[ModelTrainer]
    evaluator:     type[ModelEvaluator]
    report_gen:    type[ReportGenerator]
    description:   str = ""


class PipelineRegistry:
    """
    Central registry mapping (feature, modality) → PipelineBundle.

    Usage
    -----
    Register a new pipeline once at import time:

        from pipeline.registry import pipeline_registry
        pipeline_registry.register(PipelineBundle(
            feature="deepfake",
            modality="image",
            validator=DeepfakeImageValidator,
            preprocessor=DeepfakeImagePreprocessor,
            splitter=DeepfakeImageSplitter,
            trainer=DeepfakeImageTrainer,
            evaluator=DeepfakeImageEvaluator,
            report_gen=DeepfakeImageReportGenerator,
        ))

    The orchestrator then uses:

        bundle = pipeline_registry.get("deepfake", "image")
    """

    def __init__(self):
        self._bundles: dict[str, PipelineBundle] = {}

    def register(self, bundle: PipelineBundle) -> None:
        key = f"{bundle.feature}.{bundle.modality}"
        if key in self._bundles:
            logger.warning("Overwriting existing pipeline bundle for '%s'.", key)
        self._bundles[key] = bundle
        logger.info("Registered pipeline: %s", key)

    def get(self, feature: str, modality: str) -> PipelineBundle | None:
        return self._bundles.get(f"{feature}.{modality}")

    def list_pipelines(self) -> list[str]:
        return list(self._bundles.keys())

    def is_registered(self, feature: str, modality: str) -> bool:
        return f"{feature}.{modality}" in self._bundles


# ── Module-level singleton ────────────────────────────────────────────────────

pipeline_registry = PipelineRegistry()
