"""
CyberGuard ML Server — Pipeline Job Tracker.

Every pipeline run creates a Job that tracks state, logs, and metadata.
Jobs are persisted to the filesystem under jobs/<job_id>/.

Job lifecycle:

    QUEUED → VALIDATING → PREPROCESSING → SPLITTING
          → TRAINING → EVALUATING → GENERATING_REPORT
          → WAITING_FOR_APPROVAL → APPROVED/REJECTED → PROMOTED/FAILED
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.base import JobState

logger = logging.getLogger(__name__)

# Root directory for all job metadata
_JOBS_ROOT = Path(__file__).parent.parent / "jobs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    """
    Represents a single pipeline run (one dataset → one candidate model).

    Stored as jobs/<job_id>/job.json.
    Logs are appended to jobs/<job_id>/logs/pipeline.log.
    """
    job_id: str
    feature: str
    modality: str
    dataset_id: str
    dataset_version: str
    model_version: str
    state: JobState             = JobState.QUEUED
    created_at: str             = field(default_factory=_now_iso)
    updated_at: str             = field(default_factory=_now_iso)
    completed_at: str | None    = None
    error: str | None           = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any]    = field(default_factory=dict)

    @property
    def job_dir(self) -> Path:
        return _JOBS_ROOT / self.job_id

    @property
    def log_path(self) -> Path:
        return self.job_dir / "logs" / "pipeline.log"

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist job metadata to disk."""
        self.updated_at = _now_iso()
        self.job_dir.mkdir(parents=True, exist_ok=True)
        (self.job_dir / "logs").mkdir(parents=True, exist_ok=True)
        job_path = self.job_dir / "job.json"
        job_path.write_text(
            json.dumps(asdict(self), indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, job_id: str) -> "Job | None":
        """Load a job from disk. Returns None if not found."""
        job_path = _JOBS_ROOT / job_id / "job.json"
        if not job_path.exists():
            return None
        try:
            data = json.loads(job_path.read_text(encoding="utf-8"))
            data["state"] = JobState(data["state"])
            return cls(**data)
        except Exception as exc:
            logger.error("Failed to load job '%s': %s", job_id, exc)
            return None

    # ── State transitions ────────────────────────────────────────────────────

    def transition(self, new_state: JobState, note: str = "") -> None:
        """Transition to a new state and record the step."""
        old_state = self.state
        self.state = new_state
        step = {
            "from": old_state.value,
            "to": new_state.value,
            "timestamp": _now_iso(),
        }
        if note:
            step["note"] = note
        self.steps.append(step)
        self.log(f"State: {old_state.value} → {new_state.value}" + (f" — {note}" if note else ""))
        self.save()

    def fail(self, reason: str) -> None:
        """Mark job as FAILED with a reason."""
        self.error = reason
        self.transition(JobState.FAILED, note=reason)
        self.completed_at = _now_iso()
        self.save()

    def complete(self, final_state: JobState, note: str = "") -> None:
        """Mark job as complete."""
        self.transition(final_state, note=note)
        self.completed_at = _now_iso()
        self.save()

    # ── Logging ──────────────────────────────────────────────────────────────

    def log(self, message: str, level: str = "INFO") -> None:
        """Append a line to this job's log file."""
        try:
            (self.job_dir / "logs").mkdir(parents=True, exist_ok=True)
            ts = _now_iso()
            line = f"{ts} {level:8s} {message}\n"
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception as exc:
            logger.warning("Failed to write to job log: %s", exc)


def create_job(
    feature: str,
    modality: str,
    dataset_id: str,
    dataset_version: str,
    model_version: str,
) -> Job:
    """Create a new job, persist it, and return it."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    job_id = f"{feature}_{modality}_{model_version}_{ts}"
    job = Job(
        job_id=job_id,
        feature=feature,
        modality=modality,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        model_version=model_version,
    )
    job.save()
    logger.info("Created job '%s'.", job_id)
    return job


def list_jobs(feature: str | None = None) -> list[dict[str, Any]]:
    """List all jobs, optionally filtered by feature."""
    jobs = []
    if not _JOBS_ROOT.exists():
        return jobs
    for job_dir in sorted(_JOBS_ROOT.iterdir(), reverse=True):
        job_path = job_dir / "job.json"
        if not job_path.exists():
            continue
        try:
            data = json.loads(job_path.read_text(encoding="utf-8"))
            if feature and data.get("feature") != feature:
                continue
            jobs.append(data)
        except Exception:
            pass
    return jobs
