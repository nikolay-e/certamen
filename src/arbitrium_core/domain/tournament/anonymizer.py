"""Model anonymization for unbiased tournament evaluation."""

import random

from arbitrium_core.shared.constants import ANONYMOUS_MODEL_PREFIX
from arbitrium_core.shared.logging import get_contextual_logger


class ModelAnonymizer:
    """Handles model anonymization to prevent bias in evaluations."""

    def __init__(self, deterministic_mode: bool = False):
        self.logger = get_contextual_logger("arbitrium.anonymizer")
        self.rng = random.Random(42) if deterministic_mode else random.Random()

    @staticmethod
    def anonymize_model_keys(model_keys: list[str]) -> dict[str, str]:
        return {
            key: f"{ANONYMOUS_MODEL_PREFIX}{i + 1}"
            for i, key in enumerate(model_keys)
        }

    def anonymize_responses(
        self, responses: dict[str, str]
    ) -> tuple[dict[str, str], dict[str, str]]:
        model_names = sorted(responses.keys())
        code_names = [
            f"{ANONYMOUS_MODEL_PREFIX}{i + 1}" for i in range(len(model_names))
        ]

        forward_mapping = {
            name: code_names[i] for i, name in enumerate(model_names)
        }
        reverse_mapping = {v: k for k, v in forward_mapping.items()}

        anonymized_responses = {
            forward_mapping[name]: responses[name] for name in model_names
        }

        self.logger.debug("Anonymization mapping (alphabetical order):")
        for real_name, anon_name in forward_mapping.items():
            self.logger.debug(f"  {real_name} → {anon_name}")

        return anonymized_responses, reverse_mapping
