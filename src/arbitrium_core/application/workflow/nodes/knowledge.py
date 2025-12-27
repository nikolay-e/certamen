from typing import Any

from arbitrium_core.application.workflow.nodes.base import (
    BaseNode,
    ExecutionContext,
    Port,
    PortType,
    merge_indexed_lists,
    safe_generate,
)
from arbitrium_core.application.workflow.registry import register_node
from arbitrium_core.shared.constants import (
    INSIGHT_EXTRACTION_PROMPT,
    MAX_MULTI_INPUTS,
)
from arbitrium_core.shared.text import parse_insight_lines


def parse_insights(response_text: str) -> list[str]:
    return parse_insight_lines(
        response_text, min_length=10, skip_apologies=True
    )


@register_node
class ExtractInsightsNode(BaseNode):
    NODE_TYPE = "tournament/extract_insights"
    DISPLAY_NAME = "Extract Insights"
    CATEGORY = "Tournament"
    DESCRIPTION = "Extract key insights from eliminated model response"

    INPUTS = [
        Port(
            "response",
            PortType.STRING,
            required=True,
            description="Connect eliminated model's response to salvage good ideas from it",
        ),
        Port(
            "model",
            PortType.MODEL,
            required=False,
            description="Which LLM analyzes the response. Leave empty to use any available model",
        ),
    ]

    OUTPUTS = [
        Port(
            "insights",
            PortType.INSIGHTS,
            description="Key points extracted - connect to Inject Insights or Accumulate",
        ),
        Port(
            "insights_text",
            PortType.STRING,
            description="Same insights as readable bullet points",
        ),
    ]

    PROPERTIES = {
        "max_insights": {
            "type": "integer",
            "default": 10,
            "min": 1,
            "max": 50,
            "description": "Limit how many insights to extract (more = longer processing time)",
        },
    }

    async def execute(
        self, inputs: dict[str, Any], context: ExecutionContext
    ) -> dict[str, Any]:
        response_text = inputs.get("response", "")
        extractor_model = inputs.get("model")

        if not response_text or len(response_text) < 50:
            return {"insights": [], "insights_text": ""}

        max_insights = int(self.node_properties.get("max_insights", 10))

        if extractor_model is None:
            model_keys = list(context.models.keys())
            if not model_keys:
                return {
                    "insights": [],
                    "insights_text": "[No model available]",
                }
            extractor_model = context.models[model_keys[0]]

        prompt = INSIGHT_EXTRACTION_PROMPT.format(text=response_text)

        content, success = await safe_generate(extractor_model, prompt)
        if not success:
            return {"insights": [], "insights_text": "[Generation failed]"}

        insights = parse_insights(content)[:max_insights]
        insights_text = "\n".join(f"• {i}" for i in insights)

        return {"insights": insights, "insights_text": insights_text}


@register_node
class InjectInsightsNode(BaseNode):
    NODE_TYPE = "tournament/inject_insights"
    DISPLAY_NAME = "Inject Insights"
    CATEGORY = "Tournament"
    DESCRIPTION = "Inject Knowledge Bank insights into prompt"

    INPUTS = [
        Port(
            "prompt",
            PortType.STRING,
            required=True,
            description="The original question/task that will be enhanced with insights",
        ),
        Port(
            "insights",
            PortType.INSIGHTS,
            required=False,
            description="Connect Extract Insights or Accumulate node to add preserved knowledge",
        ),
    ]

    OUTPUTS = [
        Port(
            "enhanced_prompt",
            PortType.STRING,
            description="Original prompt + insights section - connect to Generate node",
        ),
    ]

    PROPERTIES = {
        "position": {
            "type": "select",
            "default": "before",
            "options": ["before", "after"],
            "description": "before: insights shown first as context, after: insights as reference at end",
        },
    }

    async def execute(
        self, inputs: dict[str, Any], context: ExecutionContext
    ) -> dict[str, Any]:
        prompt = inputs.get("prompt", "")
        insights = inputs.get("insights", [])

        if not insights:
            return {"enhanced_prompt": prompt}

        position = self.node_properties.get("position", "before")

        insights_lines = [
            "=== KNOWLEDGE BANK: KEY INSIGHTS FROM ELIMINATED MODELS ===",
            "Consider these preserved insights from previous analysis:",
            "",
        ]
        insights_lines.extend(f"• {insight}" for insight in insights)
        insights_lines.extend(["", "=== END KNOWLEDGE BANK ===", ""])
        insights_section = "\n".join(insights_lines) + "\n"

        if position == "before":
            enhanced = insights_section + prompt
        else:
            enhanced = prompt + "\n\n" + insights_section

        return {"enhanced_prompt": enhanced}


@register_node
class AccumulateInsightsNode(BaseNode):
    NODE_TYPE = "tournament/accumulate_insights"
    DISPLAY_NAME = "Accumulate Insights"
    CATEGORY = "Tournament"
    DESCRIPTION = "Combine insights from multiple sources"

    INPUTS = [
        Port(
            f"insights{i}",
            PortType.INSIGHTS,
            required=False,
            description=f"Connect Extract Insights from eliminated model #{i}",
        )
        for i in range(1, MAX_MULTI_INPUTS + 1)
    ]

    OUTPUTS = [
        Port(
            "combined",
            PortType.INSIGHTS,
            description="All insights merged - connect to Inject Insights node",
        ),
        Port(
            "count",
            PortType.STRING,
            description="Total number of unique insights collected",
        ),
    ]

    PROPERTIES = {
        "deduplicate": {
            "type": "boolean",
            "default": True,
            "description": "Enable to skip similar insights (recommended to avoid repetition)",
        },
    }

    async def execute(
        self, inputs: dict[str, Any], context: ExecutionContext
    ) -> dict[str, Any]:
        all_insights = merge_indexed_lists(inputs, "insights", separator="")

        deduplicate = self.node_properties.get("deduplicate", True)
        if deduplicate:
            seen = set()
            unique = []
            for insight in all_insights:
                normalized = insight.lower().strip()
                if normalized not in seen:
                    seen.add(normalized)
                    unique.append(insight)
            all_insights = unique

        return {"combined": all_insights, "count": str(len(all_insights))}
