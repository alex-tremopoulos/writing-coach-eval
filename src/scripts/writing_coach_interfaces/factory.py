from __future__ import annotations

from src.scripts.writing_coach_interfaces.base import WritingCoachInterface
from src.scripts.writing_coach_interfaces.v2 import WritingCoachV2Interface
from src.scripts.writing_coach_interfaces.v3 import WritingCoachV3Interface


def create_writing_coach(version: str) -> WritingCoachInterface:
    """Instantiate a version-specific Writing Coach adapter."""
    normalized = (version or "").strip().lower()

    if normalized == "v2":
        return WritingCoachV2Interface()
    if normalized == "v3":
        return WritingCoachV3Interface()

    raise ValueError(
        f"Unsupported Writing Coach version '{version}'. Valid options: v2, v3."
    )

