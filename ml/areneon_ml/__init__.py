"""Areneon's training pipeline and policy runtime (ADR-010)."""

import warnings

# torch looks for numpy at import. The shipped closure has none (ADR-010 §5.1) and torch
# does not need it for anything used here, so the notice is only noise on a worker's stderr.
warnings.filterwarnings("ignore", message="Failed to initialize NumPy", category=UserWarning)
