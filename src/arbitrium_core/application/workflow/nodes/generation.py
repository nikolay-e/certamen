from typing import Any

from arbitrium_core.application.workflow.nodes.base import (
    BaseNode,
    ExecutionContext,
    Port,
    PortType,
    parallel_generate,
    require_inputs,
)
from arbitrium_core.application.workflow.registry import register_node


@register_node
class GenerateNode(BaseNode):
    NODE_TYPE = "tournament/generate"
    DISPLAY_NAME = "Generate"
    CATEGORY = "Tournament"
    DESCRIPTION = "Generate responses from multiple models in parallel"
    INPUTS = [
        Port(
            "models",
            PortType.MODELS,
            description="Connect Models node here - all models will answer the same prompt",
        ),
        Port(
            "prompt",
            PortType.STRING,
            description="The question or task that all models will respond to",
        ),
    ]
    OUTPUTS = [
        Port(
            "responses",
            PortType.RESPONSES,
            description="Each model's answer, ready for evaluation or display",
        ),
    ]
    PROPERTIES = {
        "system_prompt": {
            "type": "string",
            "default": "",
            "multiline": True,
            "description": "Instructions defining LLM behavior for all models (e.g., 'You are an expert analyst' or 'Be concise and factual')",
        },
    }

    @require_inputs("models")
    async def execute(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        models, error = await self.ensure_models_or_empty(inputs["models"])
        if error:
            return error

        prompt = inputs.get("prompt", "")
        system_prompt = self.node_properties.get("system_prompt", "")

        if system_prompt:
            for model in models.values():
                if hasattr(model, "system_prompt"):
                    model.system_prompt = system_prompt

        responses = await parallel_generate(
            models,
            prompt_fn=lambda key, model: prompt,
        )
        return {"responses": responses}


@register_node
class ImproveNode(BaseNode):
    NODE_TYPE = "tournament/improve"
    DISPLAY_NAME = "Improve"
    CATEGORY = "Tournament"
    DESCRIPTION = "Improve responses based on feedback and insights"
    INPUTS = [
        Port(
            "models",
            PortType.MODELS,
            description="Same models that generated original responses",
        ),
        Port(
            "responses",
            PortType.RESPONSES,
            description="Original answers that need improvement",
        ),
        Port(
            "feedback",
            PortType.RESPONSES,
            required=False,
            description="Connect evaluations from Peer Review or Judge node to show models what to fix",
        ),
        Port(
            "insights",
            PortType.INSIGHTS,
            required=False,
            description="Good ideas from eliminated models to incorporate (from Extract Insights node)",
        ),
    ]
    OUTPUTS = [
        Port(
            "improved",
            PortType.RESPONSES,
            description="Revised answers after models addressed feedback",
        ),
    ]
    PROPERTIES = {
        "instruction": {
            "type": "string",
            "default": "Improve your answer based on the feedback.",
            "multiline": True,
            "description": "Tell models how to improve (e.g., 'Fix errors mentioned in feedback' or 'Add more detail')",
        },
    }

    @require_inputs("models", "responses")
    async def execute(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        models, error = await self.ensure_models_or_empty(inputs["models"])
        if error:
            return error

        responses = inputs["responses"]
        feedback = inputs.get("feedback", {})
        insights = inputs.get("insights", "")
        instruction = self.node_properties.get("instruction", "")

        def build_improve_prompt(model_key: str, model: Any) -> str:
            parts = [f"Original response:\n{responses.get(model_key, '')}"]

            model_feedback = feedback.get(model_key, {})
            if model_feedback:
                feedback_text = "\n".join(
                    f"- {k}: {v}" for k, v in model_feedback.items()
                )
                parts.append(f"Feedback received:\n{feedback_text}")

            if insights:
                parts.append(f"Knowledge Bank insights:\n{insights}")

            parts.append(f"Instruction: {instruction}")
            return "\n\n".join(parts)

        improved = await parallel_generate(
            models, prompt_fn=build_improve_prompt
        )
        return {"improved": improved}
