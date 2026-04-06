"""Few-shot hardware-conditioned NAS prototype package."""

from . import adaptation, backends, board_serial, datasets, hardware, io, models, paper_viz, pipeline, reporting, search, search_space
from .types import (
    ArchitectureSpec,
    BudgetSpec,
    CostPrediction,
    HardwareResponseCoefficients,
    HardwareStaticSpec,
    ProbeMeasurement,
    ReferenceMeasurement,
    StageSpec,
)

__all__ = [
    "adaptation",
    "backends",
    "board_serial",
    "datasets",
    "hardware",
    "io",
    "models",
    "paper_viz",
    "pipeline",
    "reporting",
    "search",
    "search_space",
    "ArchitectureSpec",
    "BudgetSpec",
    "CostPrediction",
    "HardwareResponseCoefficients",
    "HardwareStaticSpec",
    "ProbeMeasurement",
    "ReferenceMeasurement",
    "StageSpec",
]
