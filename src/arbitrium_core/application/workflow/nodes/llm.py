from typing import Any

import litellm

from arbitrium_core.application.workflow.nodes.base import (
    BaseNode,
    ExecutionContext,
    Port,
    PortType,
    safe_generate,
)
from arbitrium_core.application.workflow.registry import register_node
from arbitrium_core.infrastructure.config.env import get_ollama_base_url
from arbitrium_core.infrastructure.llm.factory import (
    ensure_single_model_instance,
)
from arbitrium_core.infrastructure.llm.registry import ProviderRegistry
from arbitrium_core.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)

# Non-chat model patterns to filter out when listing available models
NON_CHAT_MODEL_PATTERNS = frozenset(
    ["embedding", "whisper", "tts", "moderation", "stt", "dall-e"]
)


def _is_chat_model(model_name: str) -> bool:
    """Check if model is a chat model (not embedding, tts, etc.)."""
    name_lower = model_name.lower()
    return not any(
        pattern in name_lower for pattern in NON_CHAT_MODEL_PATTERNS
    )


def get_provider_options() -> list[str]:
    return ProviderRegistry.list_providers()


async def get_models_by_provider(provider: str) -> list[str]:
    if provider == "ollama":
        try:
            import httpx

            base_url = get_ollama_base_url()
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                models_list = data.get("models", [])
                return [f"ollama/{model['name']}" for model in models_list]
        except RuntimeError:
            return []
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama models: {e}")
            return []

    # Use LiteLLM's models_by_provider as primary source
    if hasattr(litellm, "models_by_provider"):
        provider_models = litellm.models_by_provider.get(provider, [])
        if provider_models:
            chat_models = [m for m in provider_models if _is_chat_model(m)]
            return sorted(chat_models)[:100]

    # Fallback: filter model_cost by litellm_provider field
    models = []
    for model_name, model_info in litellm.model_cost.items():
        if model_info.get("litellm_provider") == provider:
            if not _is_chat_model(model_name):
                continue
            if model_info.get("mode") in ("chat", None):
                models.append(model_name)

    return sorted(models)[:100]


@register_node
class TextNode(BaseNode):
    NODE_TYPE = "simple/text"
    DISPLAY_NAME = "Text"
    CATEGORY = "Simple"
    DESCRIPTION = (
        "Text input/output with page navigation for execution history"
    )

    INPUTS = [
        Port(
            "input_text",
            PortType.STRING,
            required=False,
            description="Connect another node's output here to pass text through dynamically",
        ),
    ]

    OUTPUTS = [
        Port(
            "output_text",
            PortType.STRING,
            description="The text value - either from input or from the text fields",
        ),
    ]

    PROPERTIES = {
        "texts": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
            "description": "Text items (one or more). Auto-adds empty field when typing.",
        },
        "separator": {
            "type": "string",
            "default": "\n",
            "description": "Separator for joining multiple texts (default: newline)",
        },
        "pages": {
            "type": "array",
            "items": {"type": "array"},
            "default": [],
            "ui_hidden": True,
            "description": "Execution history pages (internal)",
        },
        "current_page": {
            "type": "integer",
            "default": 0,
            "min": 0,
            "ui_hidden": True,
            "description": "Current page index (internal)",
        },
        "hidden": {
            "type": "boolean",
            "default": False,
            "description": "Enable to mask text display - use for API keys or sensitive data",
        },
    }

    async def execute(
        self, inputs: dict[str, Any], context: ExecutionContext
    ) -> dict[str, Any]:
        # Read pages from previous execution output if available,
        # otherwise fall back to properties (for first execution)
        if "_prev__pages" in inputs:
            pages: list[list[str]] = list(inputs["_prev__pages"])
        else:
            pages = list(self.node_properties.get("pages", []))

        if "input_text" in inputs and inputs["input_text"] is not None:
            input_data = inputs["input_text"]
            new_page = self._to_page(input_data)
            if new_page:
                pages.append(new_page)

            current_page_idx = len(pages) - 1 if pages else 0
            current_content = pages[current_page_idx] if pages else []
            output_text = (
                "\n---\n".join(current_content) if current_content else ""
            )

            return {
                "output_text": output_text,
                "_pages": pages,
                "_total_pages": len(pages),
                "_current_page": current_page_idx,
            }
        else:
            texts_array = self.node_properties.get("texts", [])
            non_empty_texts = [t for t in texts_array if t and t.strip()]
            separator = self.node_properties.get("separator", "\n")
            text = separator.join(non_empty_texts) if non_empty_texts else ""
            return {"output_text": text}

    def _to_page(self, data: Any) -> list[str]:
        if data is None:
            return []
        if isinstance(data, str):
            return [data] if data.strip() else []
        if isinstance(data, list):
            if not data:
                return []
            if isinstance(data[0], list):
                return ["\n".join(str(cell) for cell in row) for row in data]
            return [str(item) for item in data if item]
        if isinstance(data, dict):
            return [
                f"[{k}]\n{v}" if isinstance(v, str) else f"[{k}]\n{v!s}"
                for k, v in data.items()
            ]
        return [str(data)]


@register_node
class LLMNode(BaseNode):
    NODE_TYPE = "simple/llm"
    DISPLAY_NAME = "LLM"
    CATEGORY = "Simple"
    DESCRIPTION = "Universal LLM - supports any LiteLLM provider/model. Can accept model config from tournament champion."

    INPUTS = [
        Port(
            "prompt",
            PortType.STRING,
            required=True,
            description="The question or task you want the LLM to answer or perform",
        ),
        Port(
            "system",
            PortType.STRING,
            required=False,
            description="Optional role/persona for the LLM (e.g., 'You are a Python expert' or 'Answer in bullet points')",
        ),
        Port(
            "model_config",
            PortType.MODEL,
            required=False,
            description="Connect tournament champion here to reuse winning model settings",
        ),
    ]

    OUTPUTS = [
        Port(
            "response",
            PortType.STRING,
            description="Generated text answer from the LLM",
        ),
        Port(
            "model_config",
            PortType.MODEL,
            description="This model's settings - connect to Models node for tournament",
        ),
    ]

    PROPERTIES = {
        "name": {
            "type": "string",
            "default": "",
            "description": "Friendly name shown in tournament results (e.g., 'GPT-4' or 'Claude')",
        },
        "provider": {
            "type": "select",
            "default": "ollama",
            "options": get_provider_options(),
            "description": "Service hosting the model: ollama (local), openai, anthropic, etc.",
        },
        "model_name": {
            "type": "select",
            "default": "ollama/llama3.2:3b",
            "options": [],
            "description": "Specific model to use - list updates based on selected provider",
            "dynamic": True,
            "depends_on": "provider",
        },
        "temperature": {
            "type": "number",
            "default": 0.7,
            "min": 0.0,
            "max": 2.0,
            "step": 0.1,
            "description": "Controls randomness: 0 = deterministic, 1 = creative, 2 = very random",
        },
        "max_tokens": {
            "type": "integer",
            "default": 4096,
            "min": 1,
            "max": 128000,
            "description": "Maximum tokens in response (higher = longer output, more cost)",
        },
    }

    async def execute(
        self, inputs: dict[str, Any], context: ExecutionContext
    ) -> dict[str, Any]:
        prompt = inputs.get("prompt", "")
        system_prompt = inputs.get("system", "")
        input_model_config = inputs.get("model_config")

        # Use input model_config if provided, otherwise use properties
        if input_model_config and isinstance(input_model_config, dict):
            name = input_model_config.get("name", "") or self.node_id
            provider = input_model_config.get("provider", "ollama")
            model_name = input_model_config.get(
                "model_name", "ollama/llama3.2:3b"
            )
            temperature = float(input_model_config.get("temperature", 0.7))
            max_tokens = int(input_model_config.get("max_tokens", 4096))
            if not system_prompt:
                system_prompt = input_model_config.get("system_prompt", "")
        else:
            name = self.node_properties.get("name", "") or self.node_id
            provider = self.node_properties.get("provider", "ollama")
            model_name = self.node_properties.get(
                "model_name", "ollama/llama3.2:3b"
            )
            temperature = float(self.node_properties.get("temperature", 0.7))
            max_tokens = int(self.node_properties.get("max_tokens", 4096))

        model_config: dict[str, Any] = {
            "name": name,
            "provider": provider,
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "display_name": name,
        }

        if system_prompt:
            model_config["system_prompt"] = system_prompt

        if provider == "ollama":
            model_config["base_url"] = get_ollama_base_url()

        if not model_name or model_name.strip() == "":
            return {
                "response": "[Error: No model selected]",
                "model_config": model_config,
            }

        if not prompt:
            return {"response": "", "model_config": model_config}

        try:
            model = await ensure_single_model_instance(
                model_config, key=f"node_{self.node_id}"
            )
            if model is None:
                return {
                    "response": "[Error: Failed to create model]",
                    "model_config": model_config,
                }

            content, success = await safe_generate(model, prompt)
            if not success:
                logger.error(f"LLM generation failed for node {self.node_id}")
                return {
                    "response": "[Error: Generation failed]",
                    "model_config": model_config,
                }
            return {"response": content, "model_config": model_config}
        except Exception as e:
            logger.error(f"LLM node {self.node_id} error: {e}", exc_info=True)
            return {
                "response": f"[Error: {e!s}]",
                "model_config": model_config,
            }


@register_node
class TemplateNode(BaseNode):
    NODE_TYPE = "simple/template"
    DISPLAY_NAME = "Template"
    CATEGORY = "Simple"
    DESCRIPTION = "Build prompt from template with variable substitution"

    INPUTS = []

    DYNAMIC_INPUTS = {
        "prefix": "var",
        "port_type": "string",
        "min_count": 1,
    }

    OUTPUTS = [
        Port(
            "text",
            PortType.STRING,
            description="Template with all variables replaced",
        ),
    ]

    PROPERTIES = {
        "template": {
            "type": "string",
            "default": "{var_1}",
            "multiline": True,
            "description": "Template with {var_1}, {var_2}, {var_3}, etc. placeholders",
        },
    }

    async def execute(
        self, inputs: dict[str, Any], context: ExecutionContext
    ) -> dict[str, Any]:
        template = self.node_properties.get("template", "")
        result = template
        for key, value in inputs.items():
            if value is not None:
                result = result.replace(f"{{{key}}}", str(value))
        return {"text": result}
