from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class WritingCoachInterface(ABC):
    """Contract for version-specific Writing Coach adapters used by batch scripts."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Return user-facing version key (for example: v2, v3)."""

    @abstractmethod
    def run_query(self, row_id: int, query: str, document_text: str) -> Dict[str, Any]:
        """Execute one query and return a normalized batch result payload."""

    @staticmethod
    def build_result_payload(
        *,
        row_id: int,
        query: str,
        document_text: str,
        final_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build output in a single schema regardless of Writing Coach version."""
        orch_output = final_state.get("orchestrator_output", {})
        route = orch_output.get("next_action", "UNKNOWN")
        intent = final_state.get("intent", "unknown")
        reasoning = orch_output.get("reasoning", "")

        return {
            "row_id": row_id,
            "query": query,
            "input_preview": (
                document_text[:200] + "..." if len(document_text) > 200 else document_text
            ),
            "input": document_text,
            "route": route,
            "intent": intent,
            "reasoning": reasoning,
            "response": final_state.get("response", ""),
            "suggestions": final_state.get("suggestions", []),
            "references": final_state.get("references", []),
            "research_papers": final_state.get("research_papers", []),
            "segments_count": len(final_state.get("segments", [])),
            "tools_used": final_state.get("tools_used", []),
        }

