from __future__ import annotations

from typing import Any, Dict

from src.scripts.writing_coach_interfaces.base import WritingCoachInterface
from src.scripts.writing_coach_interfaces.utils import (
    install_invoke_llm_shim,
    install_learning_formatter_shim,
)


class WritingCoachV2Interface(WritingCoachInterface):
    """Writing Coach V2 adapter with all required runtime shims and graph wiring."""

    def __init__(self) -> None:
        self._preset_config: Dict[str, Any] = {}
        self._graph = None
        self._initialize()

    @property
    def version(self) -> str:
        return "v2"

    def _build_writing_coach_v2_config(self) -> Dict[str, Any]:
        from src.presets_config import get_preset_config

        copilot_config = get_preset_config("copilot_2_v4")
        copilot_models = dict(copilot_config.get("models", {}))
        copilot_prompts = dict(copilot_config.get("prompts", {}))
        copilot_parameters = dict(copilot_config.get("parameters", {}))

        return {
            "display_name": "Writing Coach V2",
            "description": "Standalone batch config for Writing Coach V2",
            "mode_indicator": "Writing Coach V2",
            "type": "writing_coach",
            "expose_in_ui": False,
            "tools": copilot_config.get("tools", {}),
            "parameters": {
                **copilot_parameters,
                "preset": "writing_coach_v2",
                "copilot_preset": "copilot_2_v4",
                "document_search_limit": copilot_parameters.get("document_search_limit", 20),
                "supports_conversation": True,
                "conversation_turn_limit": 10,
            },
            "prompts": {
                **copilot_prompts,
                "wc_orchestrator": "v2",
                "wc_respond": "v2",
                "wc_segment_analysis": "v1",
                "research_response": "v1",
                "parallel_research_transform": "v1",
                "parallel_simple_transform": "v1",
                "revision_explanation": "v1",
            },
            "models": {
                **copilot_models,
                "orchestrator": copilot_models.get("orchestrator", "gpt-5.1-chat"),
                "research_response": copilot_models.get("copilot_summary_standard", "gpt-5.1-chat"),
                "structured_transform": "gpt-5.1-chat",
                "text_transformation": "gpt-5.1-chat",
                "respond": copilot_models.get("copilot_reinterpret", "gpt-5.1-chat"),
                "segment_analysis": "gpt-5-mini",
                "search_parameter_extraction": copilot_models.get(
                    "search_parameter_extraction", "gpt-5.1-chat"
                ),
                "learnings_extraction": copilot_models.get("learnings_extraction", "gpt-5-mini"),
            },
        }

    def _install_writing_coach_v2_config(self) -> Dict[str, Any]:
        from src.presets_config import PRESET_CONFIGS

        config = PRESET_CONFIGS.get("writing_coach_v2")
        if config is None:
            config = self._build_writing_coach_v2_config()
            PRESET_CONFIGS["writing_coach_v2"] = config
        return config

    def _initialize_writing_coach_v2_only(self) -> None:
        from langgraph.graph import END, START

        from src.graph_builder import GraphBuilder
        from src.graph_nodes.writing_coach_v2_nodes import (
            output_node,
            research_response_node,
            research_transform_node,
            revision_explanation_node,
            search_node,
            search_router,
            segment_analysis_node,
            segment_analysis_router,
            simple_transform_node,
            wc_orchestrator_node,
            wc_orchestrator_router,
            wc_respond_node,
        )
        from src.graph_presets import register_preset
        from src.state_definitions import WritingCoachV2State

        builder = GraphBuilder(WritingCoachV2State)
        builder.set_display_name("Writing Coach V2")
        builder.set_description("Conversational writing coach with hybrid graph architecture")

        for name, fn in [
            ("wc2/orchestrator", wc_orchestrator_node),
            ("wc2/segment_analysis", segment_analysis_node),
            ("wc2/copilot_search", search_node),
            ("wc2/research_response", research_response_node),
            ("wc2/research_transform", research_transform_node),
            ("wc2/simple_transform", simple_transform_node),
            ("wc2/revision_explanation", revision_explanation_node),
            ("wc2/respond", wc_respond_node),
            ("wc2/output", output_node),
        ]:
            builder.register_node_function(name, fn)
            builder.add_node(name)

        builder.add_edge(START, "wc2/orchestrator")
        builder.add_conditional_edge(
            "wc2/orchestrator",
            wc_orchestrator_router,
            {
                "wc2/segment_analysis": "wc2/segment_analysis",
                "wc2/respond": "wc2/respond",
            },
        )

        builder.add_conditional_edge(
            "wc2/segment_analysis",
            segment_analysis_router,
            {
                "wc2/copilot_search": "wc2/copilot_search",
                "wc2/simple_transform": "wc2/simple_transform",
                "wc2/revision_explanation": "wc2/revision_explanation",
                "wc2/respond": "wc2/respond",
            },
        )
        builder.add_conditional_edge(
            "wc2/copilot_search",
            search_router,
            {
                "wc2/research_response": "wc2/research_response",
                "wc2/research_transform": "wc2/research_transform",
            },
        )

        builder.add_edge("wc2/research_response", "wc2/output")
        builder.add_edge("wc2/research_transform", "wc2/revision_explanation")
        builder.add_edge("wc2/simple_transform", "wc2/revision_explanation")
        builder.add_edge("wc2/revision_explanation", "wc2/output")
        builder.add_edge("wc2/respond", "wc2/output")
        builder.add_edge("wc2/output", END)

        register_preset("writing_coach_v2", builder)
        print("Writing Coach V2 preset initialized (standalone)")

    def _initialize(self) -> None:
        from src.graph_presets import get_preset

        print("Initializing Writing Coach V2...")
        install_learning_formatter_shim()
        install_invoke_llm_shim()
        self._preset_config = self._install_writing_coach_v2_config()
        self._initialize_writing_coach_v2_only()
        builder = get_preset("writing_coach_v2")
        self._graph = builder.build_without_checkpointing()

    def run_query(self, row_id: int, query: str, document_text: str) -> Dict[str, Any]:
        from src.state_definitions import WritingCoachV2State

        initial_state: WritingCoachV2State = {
            "message": query,
            "document_text": document_text,
            "selected_text": None,
            "conversation_history": [],
            "prior_references": [],
            "conversation_id": f"batch_{row_id}",
            "conversation_turn": 1,
            "streaming": False,
            "writer": None,
            "preset": "writing_coach_v2",
            "model_config": self._preset_config.get("models", {}),
            "prompt_versions": self._preset_config.get("prompts", {}),
            "parameters": self._preset_config.get("parameters", {}),
            "suggestions": [],
            "references": [],
            "research_papers": [],
        }

        final_state = self._graph.invoke(initial_state)
        return self.build_result_payload(
            row_id=row_id,
            query=query,
            document_text=document_text,
            final_state=final_state,
        )

