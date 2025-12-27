"""The Knowledge Bank for preserving insights from eliminated models."""

import uuid
from typing import TYPE_CHECKING, Any

from arbitrium_core.ports.similarity import SimilarityEngine
from arbitrium_core.shared.constants import (
    DEFAULT_MAX_INSIGHTS,
    INSIGHT_EXTRACTION_PROMPT,
    PLACEHOLDER_RESPONSES,
)
from arbitrium_core.shared.logging import get_contextual_logger
from arbitrium_core.shared.text import parse_insight_lines
from arbitrium_core.shared.validation.response import detect_apology_or_refusal

if TYPE_CHECKING:
    from arbitrium_core.domain.tournament.tournament import ModelComparison


class EnhancedKnowledgeBank:
    """A system for preserving and retrieving valuable insights from model responses.

    Especially from models that are eliminated during the tournament.
    """

    def __init__(
        self,
        comparison_instance: "ModelComparison",
        similarity_engine: SimilarityEngine,
    ):
        self.logger = get_contextual_logger("arbitrium.core.knowledge_bank")
        self.comparison = comparison_instance
        self.insights_db: dict[str, dict[str, Any]] = {}
        self.insights_vectors: Any = None
        self.insight_ids: list[str] = []
        self._lock: Any = None  # Created lazily to avoid event loop issues
        self.similarity_engine = similarity_engine

        # Get configuration from config, with fallback defaults
        kb_config = comparison_instance.config.get("knowledge_bank", {})
        self.similarity_threshold = kb_config.get("similarity_threshold", 0.75)
        self.max_insights = kb_config.get("max_insights", DEFAULT_MAX_INSIGHTS)

    def _get_lock(self) -> Any:
        """Get or create asyncio.Lock lazily to avoid event loop issues."""
        if self._lock is None:
            import asyncio

            self._lock = asyncio.Lock()
        return self._lock

    def _determine_extractor_model_key(
        self, kb_model_config: object
    ) -> str | None:
        if kb_model_config == "leader":
            extractor_model_key = getattr(
                self.comparison, "current_leader_key", None
            )
            if not extractor_model_key:
                extractor_model_key = (
                    next(iter(self.comparison.active_model_keys), None)
                    if self.comparison.active_model_keys
                    else None
                )
                self.logger.warning(
                    "Leader not determined yet, using fallback model: %s",
                    extractor_model_key,
                )
            else:
                leader_display = self.comparison.anon_mapping.get(
                    extractor_model_key, extractor_model_key
                )
                self.logger.info(
                    "Using tournament leader %s for insight extraction",
                    leader_display,
                )
            return extractor_model_key
        else:
            if isinstance(kb_model_config, str):
                return kb_model_config
            judge_model: object = self.comparison.features.get("judge_model")
            if isinstance(judge_model, str):
                return judge_model
            return (
                next(iter(self.comparison.active_model_keys), None)
                if self.comparison.active_model_keys
                else None
            )

    def _is_valid_response_for_extraction(
        self, response_text: str
    ) -> tuple[bool, str]:
        if not response_text:
            return False, "Response is empty"

        text_stripped = response_text.strip()

        # Check if response is a technical error message FIRST
        # This should be checked before length validation
        text_lower = text_stripped.lower()
        error_prefixes = ["error:", "failed:", "timeout:", "exception:"]
        for prefix in error_prefixes:
            if text_lower.startswith(prefix):
                return (
                    False,
                    f"Response is an error message (starts with '{prefix}')",
                )

        # Check minimum length (too short to contain meaningful insights)
        if len(text_stripped) < 50:
            return (
                False,
                f"Response too short ({len(text_stripped)} chars, minimum 50)",
            )

        # Check if response is apology/refusal using shared utility
        # This is a safety net - prompts should prevent this
        if detect_apology_or_refusal(response_text):
            return (
                False,
                "Response is apology/refusal (prompt should prevent this)",
            )

        # Check for obvious placeholder responses
        if text_stripped.lower() in PLACEHOLDER_RESPONSES:
            return False, f"Response is a placeholder ('{text_stripped}')"

        return True, ""

    def _parse_claims_from_response(
        self, response_content: str, extractor_model_key: str
    ) -> list[str]:
        self.logger.debug(
            "[%s] Raw insight extraction response: %s",
            extractor_model_key,
            response_content,
        )

        if detect_apology_or_refusal(response_content):
            self.logger.error(
                "[%s] Model returned apology/refusal instead of insight extraction. Response: %s",
                extractor_model_key,
                response_content,
            )
            return []

        claims = parse_insight_lines(
            response_content, min_length=10, skip_apologies=True
        )

        if not claims:
            self.logger.warning(
                "[%s] No valid insights found in response. Response: %s",
                extractor_model_key,
                response_content,
            )

        return claims

    async def extract_and_add_insights(
        self, eliminated_response: str, model_name: str, round_num: int
    ) -> None:
        # Check if Knowledge Bank extraction is enabled
        kb_config = self.comparison.config.get("knowledge_bank", {})
        kb_enabled = kb_config.get("enabled", True)

        if not kb_enabled:
            self.logger.debug(
                f"Knowledge Bank is disabled. Skipping insight extraction for {model_name}."
            )
            return

        self.logger.info(
            f"Extracting insights from eliminated model {model_name}'s response."
        )

        # Validate the response before sending to extractor
        is_valid, reason = self._is_valid_response_for_extraction(
            eliminated_response
        )
        if not is_valid:
            self.logger.debug(
                f"Skipping insight extraction for {model_name}: {reason}. Response: {eliminated_response}"
            )
            return

        prompt = INSIGHT_EXTRACTION_PROMPT.format(text=eliminated_response)

        # Determine which model should extract insights
        kb_model_config = self.comparison.features.get("knowledge_bank_model")
        extractor_model_key = self._determine_extractor_model_key(
            kb_model_config
        )

        # Validate the model exists and is available
        if (
            not extractor_model_key
            or extractor_model_key not in self.comparison.models
        ):
            self.logger.error(
                f"No available model for insight extraction. Configured: {extractor_model_key}"
            )
            return

        response = await self.comparison._execute_single_model_task(
            model_key=extractor_model_key,
            prompt=prompt,
            context_for_logging=f"insight extraction from {model_name}",
        )

        if response.is_error():
            self.logger.error(
                f"Failed to extract insights from {model_name}: {response.error}"
            )
            return

        try:
            claims = self._parse_claims_from_response(
                response.content, extractor_model_key
            )
        except Exception as e:
            self.logger.error(
                f"Failed to parse insights from LLM response: {e}"
            )
            return

        self.logger.info(
            f"Extracted {len(claims)} potential insights from {model_name}."
        )
        await self._add_insights_to_db(claims, model_name, round_num)

    def _revectorize_insights(self) -> None:
        if self.insights_db:
            all_texts = [i["text"] for i in self.insights_db.values()]
            self.insights_vectors = self.similarity_engine.transform(all_texts)
        else:
            self.insights_vectors = None

    def _ensure_vectorizer_fitted(self, claims: list[str]) -> None:
        if not self.similarity_engine.is_fitted():
            all_known_texts = [
                i["text"] for i in self.insights_db.values()
            ] + claims
            self.similarity_engine.fit(all_known_texts)

    def _is_duplicate_claim(
        self, claim_vector: object, similarity_threshold: float
    ) -> bool:
        if (
            self.insights_vectors is None
            or self.insights_vectors.shape[0] == 0
        ):
            return False
        similarity_scores = self.similarity_engine.compute_similarity(
            claim_vector, self.insights_vectors
        )
        is_duplicate = max(similarity_scores) > similarity_threshold
        return is_duplicate

    def _add_claim_to_db(
        self, claim: str, source_model: str, source_round: int
    ) -> str:
        insight_id = str(uuid.uuid4())
        self.insights_db[insight_id] = {
            "text": claim,
            "source_model": source_model,
            "source_round": source_round,
        }
        self.insight_ids.append(insight_id)
        return insight_id

    def _enforce_max_insights(self) -> None:
        if len(self.insight_ids) <= self.max_insights:
            return

        num_to_remove = len(self.insight_ids) - self.max_insights
        removed_ids = self.insight_ids[:num_to_remove]
        for insight_id in removed_ids:
            self.insights_db.pop(insight_id, None)
        self.insight_ids = self.insight_ids[num_to_remove:]
        self.logger.info(
            f"Removed {num_to_remove} oldest insights to maintain limit of {self.max_insights}"
        )
        self._revectorize_insights()

    async def _add_insights_to_db(
        self, claims: list[str], source_model: str, source_round: int
    ) -> None:
        if not claims:
            return

        async with self._get_lock():
            self._ensure_vectorizer_fitted(claims)

            if self.insights_db:
                self.insights_vectors = self.similarity_engine.transform(
                    [i["text"] for i in self.insights_db.values()]
                )

            new_vectors = self.similarity_engine.transform(claims)

            added_count = 0
            for i, claim in enumerate(claims):
                if not self._is_duplicate_claim(
                    new_vectors[i], self.similarity_threshold
                ):
                    self._add_claim_to_db(claim, source_model, source_round)
                    added_count += 1

            if added_count > 0:
                self._revectorize_insights()
                self.logger.info(
                    f"Added {added_count} new unique insights to the Knowledge Bank. Total insights: {len(self.insights_db)}"
                )
                self._enforce_max_insights()

    async def get_all_insights(self) -> list[dict[str, Any]]:
        async with self._get_lock():
            if not self.insights_db:
                return []

            return [
                self.insights_db[insight_id] for insight_id in self.insight_ids
            ]

    async def get_insights_for_model(
        self, model_name: str, round_num: int | None = None
    ) -> list[str]:
        async with self._get_lock():
            if not self.insights_db:
                return []

            insights = []
            for insight_id in self.insight_ids:
                insight_data = self.insights_db[insight_id]
                if insight_data["source_model"] == model_name:
                    if (
                        round_num is None
                        or insight_data["source_round"] == round_num
                    ):
                        insights.append(insight_data["text"])

            return insights

    async def format_insights_for_context(self) -> str:
        kb_config = self.comparison.config.get("knowledge_bank", {})
        kb_enabled = kb_config.get("enabled", True)

        if not kb_enabled:
            return ""

        insights = await self.get_all_insights()
        if not insights:
            return ""

        header = (
            "\n=== KNOWLEDGE BANK: KEY INSIGHTS FROM ELIMINATED MODELS ===\n"
        )
        hint = "Keep these facts in mind - they may contain valuable perspectives:\n\n"
        formatted_insights = "\n".join(
            f"• [{insight['source_model']}, Round {insight['source_round']}]: {insight['text']}"
            for insight in insights
        )
        footer = "\n=== END KNOWLEDGE BANK ===\n"
        return f"{header}{hint}{formatted_insights}{footer}"
