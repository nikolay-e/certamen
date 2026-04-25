from typing import Any

from certamen.application.workflow.nodes.base import (
    BaseNode,
    ExecutionContext,
    Port,
    PortType,
    safe_generate,
)
from certamen.application.workflow.registry import register_node
from certamen.domain.knowledge_map.builder import KnowledgeMapBuilder
from certamen.domain.knowledge_map.renderer import KnowledgeMapRenderer
from certamen.domain.prompts.builder import PromptBuilder
from certamen.domain.prompts.formatter import PromptFormatter
from certamen.infrastructure.config.defaults import get_defaults


def _build_prompt_builder(
    instruction: str | None = None,
) -> PromptBuilder:
    defaults = get_defaults()
    prompts = defaults["prompts"]
    if instruction:
        prompts = {**prompts, "synthesis": {"content": instruction}}
    return PromptBuilder(prompts, PromptFormatter())


@register_node
class SynthesizeNode(BaseNode):
    NODE_TYPE = "tournament/synthesize"
    DISPLAY_NAME = "Synthesize"
    CATEGORY = "Tournament"
    DESCRIPTION = (
        "Synthesize a single comprehensive answer from all responses, "
        "preserving consensus, contested points, and unique findings."
    )

    INPUTS = [
        Port(
            "responses",
            PortType.RESPONSES,
            required=True,
            description="All final responses from generators (Generate or Improve node).",
        ),
        Port(
            "question",
            PortType.STRING,
            required=True,
            description="Original question being answered.",
        ),
        Port(
            "model",
            PortType.MODEL,
            required=False,
            description="Model that performs the synthesis. If empty, uses first available.",
        ),
        Port(
            "kb_context",
            PortType.STRING,
            required=False,
            description="Optional knowledge bank text to include in synthesis context.",
        ),
    ]

    OUTPUTS = [
        Port(
            "synthesis",
            PortType.STRING,
            description="Unified synthesized answer combining all perspectives.",
        ),
    ]

    PROPERTIES = {
        "instruction": {
            "type": "string",
            "default": "",
            "description": (
                "Optional override for synthesis instruction. "
                "Leave empty to use the default Diamond synthesis prompt."
            ),
        },
    }

    async def execute(
        self, inputs: dict[str, Any], context: ExecutionContext
    ) -> dict[str, Any]:
        responses = inputs.get("responses") or {}
        question = inputs.get("question", "")
        kb_context = inputs.get("kb_context", "") or ""
        synth_model = inputs.get("model")

        if not responses or not question:
            return {"synthesis": ""}

        if synth_model is None:
            model_keys = list(context.models.keys())
            if not model_keys:
                return {"synthesis": "[No model available for synthesis]"}
            synth_model = context.models[model_keys[0]]

        instruction = self.node_properties.get("instruction") or None
        prompt_builder = _build_prompt_builder(instruction)
        synthesis_prompt = prompt_builder.build_synthesis_prompt(
            initial_question=question,
            all_responses=dict(responses),
            kb_context=kb_context,
        )

        content, success = await safe_generate(synth_model, synthesis_prompt)
        if not success or not content.strip():
            return {"synthesis": "[Synthesis failed]"}

        return {"synthesis": content.strip()}


@register_node
class KnowledgeMapNode(BaseNode):
    NODE_TYPE = "tournament/knowledge_map"
    DISPLAY_NAME = "Knowledge Map"
    CATEGORY = "Tournament"
    DESCRIPTION = (
        "Build a structured knowledge map: consensus claims, unique insights, "
        "known unknowns, assumptions, and exploration branches."
    )

    INPUTS = [
        Port(
            "responses",
            PortType.RESPONSES,
            required=True,
            description="All model responses to analyze.",
        ),
        Port(
            "question",
            PortType.STRING,
            required=True,
            description="Original question.",
        ),
        Port(
            "synthesis",
            PortType.STRING,
            required=False,
            description="Optional synthesis text to embed in the map.",
        ),
        Port(
            "champion",
            PortType.MODEL,
            required=False,
            description="Champion model to record in the map.",
        ),
        Port(
            "judge",
            PortType.MODEL,
            required=False,
            description="Model used to detect consensus/unique insights. Defaults to first available.",
        ),
    ]

    OUTPUTS = [
        Port(
            "knowledge_map",
            PortType.ANY,
            description="Structured KnowledgeMap object (consensus, disagreements, unique, assumptions).",
        ),
        Port(
            "markdown",
            PortType.STRING,
            description="Rendered Markdown view of the knowledge map.",
        ),
    ]

    PROPERTIES = {
        "include_exploration_branches": {
            "type": "boolean",
            "default": True,
            "description": "Generate follow-up exploration questions from the map.",
        },
    }

    async def execute(
        self, inputs: dict[str, Any], context: ExecutionContext
    ) -> dict[str, Any]:
        responses = inputs.get("responses") or {}
        question = inputs.get("question", "")
        synthesis = inputs.get("synthesis", "") or ""
        champion = inputs.get("champion")
        judge = inputs.get("judge")

        if not responses or not question:
            return {"knowledge_map": None, "markdown": ""}

        if judge is None:
            model_keys = list(context.models.keys())
            if not model_keys:
                return {
                    "knowledge_map": None,
                    "markdown": "[No model available for knowledge map]",
                }
            judge = context.models[model_keys[0]]

        builder = KnowledgeMapBuilder()
        champion_label = ""
        if champion is not None:
            champion_label = getattr(champion, "display_name", None) or str(
                champion
            )

        try:
            km = await builder.build(
                question=question,
                all_responses=dict(responses),
                synthesis=synthesis,
                champion_model=champion_label,
                judge_model=judge,
                disagreements=None,
            )
        except Exception as exc:
            return {
                "knowledge_map": None,
                "markdown": f"[Knowledge map build failed: {exc}]",
            }

        if self.node_properties.get("include_exploration_branches", True):
            try:
                km.exploration_branches = (
                    await builder.generate_exploration_branches(km, judge)
                )
            except Exception:
                km.exploration_branches = []

        try:
            markdown = KnowledgeMapRenderer().to_markdown(km)
        except Exception:
            markdown = ""

        return {"knowledge_map": km, "markdown": markdown}
