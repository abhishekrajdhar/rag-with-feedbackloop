"""MLflow tracking integration."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from core.config import Settings


class TrackingService:
    """Tracks query runs and serves compact experiment summaries."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        self.client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    def log_query_run(
        self,
        query_id: str,
        question: str,
        top_k: int,
        latency: float,
        hallucination_score: float,
        verdict: str,
        retried: bool,
    ) -> None:
        """Log a single query execution into MLflow."""
        try:
            with mlflow.start_run(run_name=query_id):
                mlflow.log_param("query_id", query_id)
                mlflow.log_param("question", question)
                mlflow.log_param("top_k", top_k)
                mlflow.log_metric("latency", latency)
                mlflow.log_metric("hallucination_score", hallucination_score)
                mlflow.log_param("verdict", verdict)
                mlflow.log_param("retried", retried)
        except Exception:
            # Tracking should not break query serving if the MLflow backend is unavailable.
            return

    def get_summary(self) -> Dict:
        """Summarize recent MLflow runs for the experiment."""
        try:
            experiment = self.client.get_experiment_by_name(self.settings.mlflow_experiment_name)
        except Exception:
            return {
                "experiment_name": self.settings.mlflow_experiment_name,
                "total_runs": 0,
                "average_latency": 0.0,
                "average_hallucination_score": 0.0,
                "verdict_counts": {},
                "recent_runs": [],
            }
        if experiment is None:
            return {
                "experiment_name": self.settings.mlflow_experiment_name,
                "total_runs": 0,
                "average_latency": 0.0,
                "average_hallucination_score": 0.0,
                "verdict_counts": {},
                "recent_runs": [],
            }

        try:
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=100,
                order_by=["attributes.start_time DESC"],
            )
        except Exception:
            return {
                "experiment_name": self.settings.mlflow_experiment_name,
                "total_runs": 0,
                "average_latency": 0.0,
                "average_hallucination_score": 0.0,
                "verdict_counts": {},
                "recent_runs": [],
            }
        if not runs:
            return {
                "experiment_name": self.settings.mlflow_experiment_name,
                "total_runs": 0,
                "average_latency": 0.0,
                "average_hallucination_score": 0.0,
                "verdict_counts": {},
                "recent_runs": [],
            }

        latencies: List[float] = []
        scores: List[float] = []
        verdicts: Counter = Counter()
        recent_runs: List[Dict] = []

        for run in runs:
            metrics = run.data.metrics
            params = run.data.params
            latency = float(metrics.get("latency", 0.0))
            score = float(metrics.get("hallucination_score", 0.0))
            verdict = params.get("verdict", "UNKNOWN")
            latencies.append(latency)
            scores.append(score)
            verdicts[verdict] += 1
            recent_runs.append(
                {
                    "run_id": run.info.run_id,
                    "query_id": params.get("query_id"),
                    "top_k": int(params.get("top_k", 0)),
                    "latency": latency,
                    "hallucination_score": score,
                    "verdict": verdict,
                    "retried": params.get("retried", "False"),
                }
            )

        return {
            "experiment_name": self.settings.mlflow_experiment_name,
            "total_runs": len(runs),
            "average_latency": round(sum(latencies) / len(latencies), 4),
            "average_hallucination_score": round(sum(scores) / len(scores), 4),
            "verdict_counts": dict(verdicts),
            "recent_runs": recent_runs[:10],
        }
