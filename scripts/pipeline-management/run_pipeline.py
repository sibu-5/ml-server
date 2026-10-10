"""
run_pipeline.py — CyberGuard ML Training Pipeline Runner.

This script runs the complete ML training pipeline for any registered feature.
It is the primary entry point for training new candidate models.

Usage
-----
    # Train deepfake image model (v5 candidate)
    python run_pipeline.py --feature deepfake --modality image --version v5

    # Train with custom config
    python run_pipeline.py --feature deepfake --modality image --version v5 \\
        --epochs 30 --batch_size 16 --lr 0.0001 --seed 42

    # Approve and promote a candidate
    python run_pipeline.py --approve --feature deepfake --modality image \\
        --version v5 --promote

    # Roll back to a previous version
    python run_pipeline.py --rollback --feature deepfake --modality image \\
        --version v4

    # List all jobs
    python run_pipeline.py --list-jobs

    # Check job status
    python run_pipeline.py --job-status deepfake_image_v5_20261009T030000

Notes
-----
- Raw data MUST exist in: data/raw/<feature>/<modality>/real/ and .../fake/
- This script NEVER overwrites the production model automatically.
- After training, review the report, then use --approve --promote to deploy.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CyberGuard ML Training Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--approve",    action="store_true", help="Approve a candidate model")
    mode_group.add_argument("--rollback",   action="store_true", help="Roll back to an older version")
    mode_group.add_argument("--list-jobs",  action="store_true", help="List all pipeline jobs")
    mode_group.add_argument("--job-status", metavar="JOB_ID",   help="Show status of a specific job")
    mode_group.add_argument("--list-pipelines", action="store_true", help="List registered pipelines")

    # ── Pipeline parameters ───────────────────────────────────────────────────
    parser.add_argument("--feature",    default="deepfake", help="Feature name (default: deepfake)")
    parser.add_argument("--modality",   default="image",    help="Modality (default: image)")
    parser.add_argument("--dataset-id", default="",         help="Dataset ID (auto-generated if empty)")
    parser.add_argument("--dataset-version", default="v1",  help="Dataset version (default: v1)")
    parser.add_argument("--version",    default="v5",       help="Target model version (default: v5)")

    # ── Training hyperparameters ──────────────────────────────────────────────
    parser.add_argument("--epochs",     type=int,   default=20,    help="Training epochs")
    parser.add_argument("--batch-size", type=int,   default=32,    help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-4,  help="Learning rate")
    parser.add_argument("--seed",       type=int,   default=42,    help="Random seed")

    # ── Approval flags ────────────────────────────────────────────────────────
    parser.add_argument("--promote",    action="store_true",
                        help="When approving, also promote to production")
    parser.add_argument("--reason",     default="",
                        help="Rejection reason (used with --reject)")
    parser.add_argument("--reject",     action="store_true",
                        help="Reject a candidate (use with --feature, --modality, --version)")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ── Import pipeline (triggers registration) ───────────────────────────────
    try:
        from pipeline import (
            PipelineOrchestrator,
            approve_model,
            reject_model,
            rollback_model,
        )
        from pipeline.job_tracker import list_jobs, Job
        from pipeline.base import pipeline_registry
    except ImportError as exc:
        logger.error("Failed to import pipeline: %s", exc)
        return 1

    # ── List pipelines ────────────────────────────────────────────────────────
    if args.list_pipelines:
        print("\nRegistered ML Pipelines:")
        for key in pipeline_registry.list_pipelines():
            print(f"  * {key}")
        return 0

    # ── List jobs ─────────────────────────────────────────────────────────────
    if args.list_jobs:
        jobs = list_jobs(feature=args.feature if args.feature else None)
        if not jobs:
            print("No jobs found.")
        for j in jobs:
            print(
                f"  {j['job_id']:50s}  state={j['state']:25s}  "
                f"created={j['created_at'][:19]}"
            )
        return 0

    # ── Job status ────────────────────────────────────────────────────────────
    if args.job_status:
        job = Job.load(args.job_status)
        if job is None:
            print(f"Job '{args.job_status}' not found.")
            return 1
        print(json.dumps({
            "job_id": job.job_id,
            "state":  job.state.value,
            "feature": job.feature,
            "modality": job.modality,
            "model_version": job.model_version,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            "error": job.error,
            "steps": job.steps,
        }, indent=2))
        return 0

    # ── Approve ───────────────────────────────────────────────────────────────
    if args.approve:
        result = approve_model(
            args.feature,
            args.modality,
            args.version,
            promote_to_production=args.promote,
        )
        print(json.dumps(result, indent=2))
        if result.get("success"):
            if args.promote:
                logger.info(
                    "[PROMOTED] Model %s/%s v%s promoted to PRODUCTION. "
                    "Restart the ML server to activate.",
                    args.feature, args.modality, args.version,
                )
            else:
                logger.info(
                    "[APPROVED] Model approved. Use --approve --promote to deploy to production."
                )
        return 0 if result.get("success") else 1

    # ── Reject ────────────────────────────────────────────────────────────────
    if args.reject:
        result = reject_model(args.feature, args.modality, args.version, reason=args.reason)
        print(json.dumps(result, indent=2))
        return 0 if result.get("success") else 1

    # ── Rollback ──────────────────────────────────────────────────────────────
    if args.rollback:
        result = rollback_model(args.feature, args.modality, args.version)
        print(json.dumps(result, indent=2))
        if result.get("success"):
            logger.info(
                "[ROLLBACK] Rolled back to %s/%s v%s. Restart the server to activate.",
                args.feature, args.modality, args.version,
            )
        return 0 if result.get("success") else 1

    # ── Run pipeline ──────────────────────────────────────────────────────────
    dataset_id = args.dataset_id or f"{args.feature}_{args.modality}_dataset"

    logger.info(
        "=== Pipeline: feature=%s modality=%s dataset=%s target_version=%s ===",
        args.feature, args.modality, dataset_id, args.version,
    )

    config = {
        "epochs":     args.epochs,
        "batch_size": args.batch_size,
        "lr":         args.lr,
        "seed":       args.seed,
    }

    orchestrator = PipelineOrchestrator()
    job = orchestrator.run(
        feature=args.feature,
        modality=args.modality,
        dataset_id=dataset_id,
        dataset_version=args.dataset_version,
        model_version=args.version,
        config=config,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Pipeline job completed.")
    print(f"  Job ID:   {job.job_id}")
    print(f"  State:    {job.state.value}")
    print(f"  Feature:  {job.feature} / {job.modality}")
    print(f"  Version:  {job.model_version}")

    if job.error:
        print(f"  Error:    {job.error}")

    report_path = job.metadata.get("report_path")
    if report_path:
        print(f"  Report:   {report_path}")

    if job.state.value == "WAITING_FOR_APPROVAL":
        print(f"\nNext step -- review the report, then run:")
        print(f"  python run_pipeline.py --approve --feature {args.feature} "
              f"--modality {args.modality} --version {args.version} --promote")
    elif job.state.value == "FAILED":
        print(f"\n[FAILED] Pipeline FAILED. Check logs at: {job.log_path}")
        return 1

    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
