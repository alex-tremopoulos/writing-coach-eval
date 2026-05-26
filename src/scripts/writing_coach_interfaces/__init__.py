from src.scripts.writing_coach_interfaces.base import WritingCoachInterface
from src.scripts.writing_coach_interfaces.factory import create_writing_coach
from src.scripts.writing_coach_interfaces.v2 import WritingCoachV2Interface
from src.scripts.writing_coach_interfaces.v3 import WritingCoachV3Interface

__all__ = [
    "WritingCoachInterface",
    "WritingCoachV2Interface",
    "WritingCoachV3Interface",
    "create_writing_coach",
]

