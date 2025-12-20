"""Experiment Logger for PopAgent.

Provides structured logging of agent decisions, method selections,
and learning progress in JSON Lines format for analysis and visualization.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid


# =============================================================================
# Log Data Classes
# =============================================================================

@dataclass
class AgentDecisionLog:
    """Log of a single agent's decision in an iteration."""
    timestamp: str
    iteration: int
    agent_id: str
    role: str
    methods_available: List[str]
    methods_selected: List[str]
    selection_scores: Dict[str, float]  # UCB/Thompson scores per method
    preferences: Dict[str, float]  # Current preference values
    context: Dict[str, Any]
    reasoning: Optional[str] = None  # LLM explanation if available
    exploration_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResultLog:
    """Log of a single pipeline evaluation."""
    pipeline_id: str
    agents: Dict[str, str]  # role -> agent_id
    methods: Dict[str, List[str]]  # role -> methods used
    pnl: float
    sharpe: float
    success: bool
    execution_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransferLog:
    """Log of a knowledge transfer event."""
    timestamp: str
    role: str
    source_agent_id: str
    target_agent_ids: List[str]
    transfer_tau: float
    methods_transferred: List[str]
    source_preferences: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiversityMetrics:
    """Diversity metrics for a population."""
    role: str
    selection_diversity: float  # 0-1, how diverse are method selections
    preference_entropy: float  # Entropy of preference distribution
    unique_methods_used: int
    total_methods_available: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IterationLog:
    """Complete log of one iteration."""
    experiment_id: str
    iteration: int
    timestamp: str
    market_regime: str
    market_context: Dict[str, Any]

    # Agent decisions
    agent_decisions: List[AgentDecisionLog]

    # Pipeline results
    pipeline_results: List[PipelineResultLog]

    # Winner info
    best_pipeline_id: Optional[str]
    best_pnl: float
    avg_pnl: float

    # Transfer info
    knowledge_transfer: Optional[TransferLog]

    # Diversity
    diversity_metrics: Dict[str, DiversityMetrics]

    # Timing
    iteration_duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "experiment_id": self.experiment_id,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "market_regime": self.market_regime,
            "market_context": self.market_context,
            "agent_decisions": [d.to_dict() for d in self.agent_decisions],
            "pipeline_results": [p.to_dict() for p in self.pipeline_results],
            "best_pipeline_id": self.best_pipeline_id,
            "best_pnl": self.best_pnl,
            "avg_pnl": self.avg_pnl,
            "knowledge_transfer": self.knowledge_transfer.to_dict() if self.knowledge_transfer else None,
            "diversity_metrics": {k: v.to_dict() for k, v in self.diversity_metrics.items()},
            "iteration_duration_ms": self.iteration_duration_ms,
        }
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class ExperimentSummary:
    """Summary of an entire experiment run."""
    experiment_id: str
    start_time: str
    end_time: str
    total_iterations: int
    config: Dict[str, Any]

    # Performance
    final_best_pnl: float
    avg_pnl_first_10: float
    avg_pnl_last_10: float
    improvement: float

    # Learning
    best_methods_by_role: Dict[str, List[str]]
    method_popularity: Dict[str, Dict[str, float]]
    transfer_count: int

    # Diversity
    final_diversity: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# Experiment Logger
# =============================================================================

class ExperimentLogger:
    """
    Logs experiment data in JSON Lines format for analysis and visualization.

    Usage:
        logger = ExperimentLogger(experiment_id="exp_001", log_dir="logs/experiments")

        for iteration in range(100):
            # ... run iteration ...
            logger.log_iteration(iteration_log)

        logger.finalize()
    """

    def __init__(
        self,
        experiment_id: Optional[str] = None,
        log_dir: Union[str, Path] = "logs/experiments",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the experiment logger.

        Args:
            experiment_id: Unique experiment identifier (auto-generated if None)
            log_dir: Directory for log files
            config: Experiment configuration to log
        """
        self.experiment_id = experiment_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or {}
        self.start_time = datetime.now(tz=timezone.utc)

        # Log files
        self.iterations_file = self.log_dir / f"{self.experiment_id}_iterations.jsonl"
        self.summary_file = self.log_dir / f"{self.experiment_id}_summary.json"

        # In-memory tracking
        self.iteration_count = 0
        self.all_pnls: List[float] = []
        self.transfer_count = 0
        self.best_methods: Dict[str, List[str]] = {}
        self.method_usage: Dict[str, Dict[str, int]] = {}

        # Write initial metadata
        self._write_metadata()

    def _write_metadata(self):
        """Write experiment metadata to a separate file."""
        metadata = {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time.isoformat(),
            "config": self.config,
        }

        metadata_file = self.log_dir / f"{self.experiment_id}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def log_iteration(self, iteration_log: IterationLog):
        """
        Log a complete iteration.

        Args:
            iteration_log: Complete iteration log
        """
        # Write to JSONL file
        with open(self.iterations_file, "a") as f:
            f.write(iteration_log.to_json() + "\n")

        # Update in-memory tracking
        self.iteration_count += 1
        self.all_pnls.append(iteration_log.best_pnl)

        if iteration_log.knowledge_transfer:
            self.transfer_count += 1

        # Track method usage
        for decision in iteration_log.agent_decisions:
            role = decision.role
            if role not in self.method_usage:
                self.method_usage[role] = {}

            for method in decision.methods_selected:
                self.method_usage[role][method] = self.method_usage[role].get(method, 0) + 1

    def log_agent_decision(
        self,
        iteration: int,
        agent_id: str,
        role: str,
        methods_available: List[str],
        methods_selected: List[str],
        selection_scores: Dict[str, float],
        preferences: Dict[str, float],
        context: Dict[str, Any],
        reasoning: Optional[str] = None,
        exploration_used: bool = False,
    ) -> AgentDecisionLog:
        """
        Create an agent decision log.

        This is a helper method to create AgentDecisionLog instances.
        """
        return AgentDecisionLog(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            iteration=iteration,
            agent_id=agent_id,
            role=role,
            methods_available=methods_available,
            methods_selected=methods_selected,
            selection_scores=selection_scores,
            preferences=preferences,
            context=context,
            reasoning=reasoning,
            exploration_used=exploration_used,
        )

    def log_pipeline_result(
        self,
        agents: Dict[str, str],
        methods: Dict[str, List[str]],
        pnl: float,
        sharpe: float,
        success: bool,
        execution_time_ms: Optional[float] = None,
    ) -> PipelineResultLog:
        """
        Create a pipeline result log.
        """
        return PipelineResultLog(
            pipeline_id=f"pipe_{uuid.uuid4().hex[:8]}",
            agents=agents,
            methods=methods,
            pnl=pnl,
            sharpe=sharpe,
            success=success,
            execution_time_ms=execution_time_ms,
        )

    def log_transfer(
        self,
        role: str,
        source_agent_id: str,
        target_agent_ids: List[str],
        transfer_tau: float,
        methods_transferred: List[str],
        source_preferences: Dict[str, float],
    ) -> TransferLog:
        """
        Create a knowledge transfer log.
        """
        return TransferLog(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            role=role,
            source_agent_id=source_agent_id,
            target_agent_ids=target_agent_ids,
            transfer_tau=transfer_tau,
            methods_transferred=methods_transferred,
            source_preferences=source_preferences,
        )

    def create_iteration_log(
        self,
        iteration: int,
        market_context: Dict[str, Any],
        agent_decisions: List[AgentDecisionLog],
        pipeline_results: List[PipelineResultLog],
        best_pipeline_id: Optional[str],
        best_pnl: float,
        avg_pnl: float,
        knowledge_transfer: Optional[TransferLog],
        diversity_metrics: Dict[str, DiversityMetrics],
        iteration_duration_ms: float,
    ) -> IterationLog:
        """
        Create a complete iteration log.
        """
        return IterationLog(
            experiment_id=self.experiment_id,
            iteration=iteration,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            market_regime=market_context.get("regime", "unknown"),
            market_context=market_context,
            agent_decisions=agent_decisions,
            pipeline_results=pipeline_results,
            best_pipeline_id=best_pipeline_id,
            best_pnl=best_pnl,
            avg_pnl=avg_pnl,
            knowledge_transfer=knowledge_transfer,
            diversity_metrics=diversity_metrics,
            iteration_duration_ms=iteration_duration_ms,
        )

    def finalize(self, best_methods: Optional[Dict[str, List[str]]] = None) -> ExperimentSummary:
        """
        Finalize the experiment and write summary.

        Args:
            best_methods: Final best methods by role

        Returns:
            ExperimentSummary object
        """
        end_time = datetime.now(tz=timezone.utc)

        # Calculate metrics
        avg_pnl_first_10 = sum(self.all_pnls[:10]) / min(10, len(self.all_pnls)) if self.all_pnls else 0.0
        avg_pnl_last_10 = sum(self.all_pnls[-10:]) / min(10, len(self.all_pnls)) if self.all_pnls else 0.0
        improvement = avg_pnl_last_10 - avg_pnl_first_10

        # Calculate method popularity
        method_popularity = {}
        for role, usage in self.method_usage.items():
            total = sum(usage.values())
            if total > 0:
                method_popularity[role] = {m: c / total for m, c in usage.items()}
            else:
                method_popularity[role] = {}

        # Create summary
        summary = ExperimentSummary(
            experiment_id=self.experiment_id,
            start_time=self.start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_iterations=self.iteration_count,
            config=self.config,
            final_best_pnl=max(self.all_pnls) if self.all_pnls else 0.0,
            avg_pnl_first_10=avg_pnl_first_10,
            avg_pnl_last_10=avg_pnl_last_10,
            improvement=improvement,
            best_methods_by_role=best_methods or self.best_methods,
            method_popularity=method_popularity,
            transfer_count=self.transfer_count,
            final_diversity={},  # Would need to track this
        )

        # Write summary
        with open(self.summary_file, "w") as f:
            f.write(summary.to_json())

        return summary

    def get_iterations(self) -> List[Dict[str, Any]]:
        """
        Load all logged iterations from file.

        Returns:
            List of iteration dictionaries
        """
        iterations = []
        if self.iterations_file.exists():
            with open(self.iterations_file, "r") as f:
                for line in f:
                    if line.strip():
                        iterations.append(json.loads(line))
        return iterations

    def get_iteration(self, iteration_num: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific iteration by number.

        Args:
            iteration_num: Iteration number (1-indexed)

        Returns:
            Iteration dictionary or None
        """
        iterations = self.get_iterations()
        for it in iterations:
            if it.get("iteration") == iteration_num:
                return it
        return None


# =============================================================================
# Utility Functions
# =============================================================================

def load_experiment(log_dir: Union[str, Path], experiment_id: str) -> Dict[str, Any]:
    """
    Load a complete experiment from log files.

    Args:
        log_dir: Directory containing log files
        experiment_id: Experiment ID to load

    Returns:
        Dictionary with metadata, iterations, and summary
    """
    log_dir = Path(log_dir)

    result = {
        "experiment_id": experiment_id,
        "metadata": None,
        "iterations": [],
        "summary": None,
    }

    # Load metadata
    metadata_file = log_dir / f"{experiment_id}_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            result["metadata"] = json.load(f)

    # Load iterations
    iterations_file = log_dir / f"{experiment_id}_iterations.jsonl"
    if iterations_file.exists():
        with open(iterations_file, "r") as f:
            for line in f:
                if line.strip():
                    result["iterations"].append(json.loads(line))

    # Load summary
    summary_file = log_dir / f"{experiment_id}_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            result["summary"] = json.load(f)

    return result


def list_experiments(log_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    List all experiments in a log directory.

    Args:
        log_dir: Directory containing log files

    Returns:
        List of experiment summaries
    """
    log_dir = Path(log_dir)
    experiments = []

    if not log_dir.exists():
        return experiments

    # Find all metadata files
    for metadata_file in log_dir.glob("*_metadata.json"):
        exp_id = metadata_file.stem.replace("_metadata", "")

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Check for summary
        summary_file = log_dir / f"{exp_id}_summary.json"
        summary = None
        if summary_file.exists():
            with open(summary_file, "r") as f:
                summary = json.load(f)

        experiments.append({
            "experiment_id": exp_id,
            "start_time": metadata.get("start_time"),
            "config": metadata.get("config", {}),
            "completed": summary is not None,
            "total_iterations": summary.get("total_iterations") if summary else None,
        })

    # Sort by start time
    experiments.sort(key=lambda x: x.get("start_time", ""), reverse=True)

    return experiments
