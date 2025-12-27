from typing import Any

from pydantic import BaseModel, Field


class RetryConfig(BaseModel):
    max_attempts: int = Field(default=3, ge=1)
    initial_delay: float = Field(default=10.0, ge=0.1)
    max_delay: float = Field(default=60.0, ge=1.0)


class FeaturesConfig(BaseModel):
    save_reports_to_disk: bool = True
    deterministic_mode: bool = True
    judge_model: str | None = None
    knowledge_bank_model: str = "leader"
    llm_compression: bool = False
    compression_model: str | None = None


class KnowledgeBankConfig(BaseModel):
    enabled: bool = True
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    max_insights: int = Field(default=100, ge=1)


class PromptMetadata(BaseModel):
    version: str = "1.0"
    type: str = "instruction"
    phase: str


class PromptConfig(BaseModel):
    content: str
    metadata: PromptMetadata


class PromptsConfig(BaseModel):
    initial: PromptConfig
    feedback: PromptConfig
    improvement: PromptConfig
    evaluate: PromptConfig


class ModelConfig(BaseModel):
    model_config = {"extra": "allow"}

    provider: str
    model_name: str
    display_name: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = None
    context_window: int | None = None
    reasoning_effort: str | None = None
    llm_compression: bool | None = None
    compression_model: str | None = None


class ArbitriumConfig(BaseModel):
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    prompts: dict[str, Any] = Field(default_factory=dict)
    knowledge_bank: KnowledgeBankConfig = Field(
        default_factory=KnowledgeBankConfig
    )
    outputs_dir: str | None = None
    question: str | None = None
