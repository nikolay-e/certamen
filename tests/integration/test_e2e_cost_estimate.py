from pathlib import Path

import pytest
import yaml

from certamen.application.cost.estimator import (
    OUTPUT_PRIORS,
    estimate_cost,
    resolve_price,
)
from certamen.application.slim_loader import materialize_workflow
from certamen.domain.errors import ConfigurationError
from certamen.infrastructure.config.slim import SlimConfig


def _ollama_diamond_config() -> SlimConfig:
    return SlimConfig(
        question="What is the capital of France?",
        models={
            k: {"provider": "ollama", "model_name": f"ollama/{k}"}
            for k in ("a", "b", "c", "d")
        },
        workflow="diamond-tournament",
    )


def _flagship_diamond_config() -> SlimConfig:
    models = {
        "gpt": {"provider": "openai", "model_name": "gpt-5.6-sol"},
        "claude": {
            "provider": "anthropic",
            "model_name": "anthropic/claude-fable-5",
        },
        "gemini": {
            "provider": "google",
            "model_name": "gemini/gemini-3.1-pro-preview",
        },
        "grok": {"provider": "xai", "model_name": "xai/grok-4.5"},
    }
    return SlimConfig(
        question="Explain submodular maximization under a matroid constraint.",
        models=models,
        workflow="diamond-tournament",
    )


class TestCallModel:
    def test_diamond_four_models_matches_empirical_call_count(self) -> None:
        # The empirical ~143-144 calls / 4 iterations was measured with a single
        # interrogation round; pin rounds=1 to keep that anchor honest (the
        # shipped default is now rounds=3, exercised in the scaling test below).
        slim = _ollama_diamond_config()
        slim.overrides["interrogate.rounds"] = 1
        workflow = materialize_workflow(slim)
        est = estimate_cost(workflow, slim.price_overrides)
        assert est.n_competitors == 4
        assert est.divergence_iterations == 4
        assert est.total_calls == 143
        # Stalled (scores never parse) runs to max_rounds > healthy.
        assert est.stalled_total_calls > est.total_calls

    def test_interrogation_rounds_scale_call_count(self) -> None:
        base = _ollama_diamond_config()
        base.overrides["interrogate.rounds"] = 1
        one = estimate_cost(materialize_workflow(base), base.price_overrides)

        three = _ollama_diamond_config()
        three.overrides["interrogate.rounds"] = 3
        est3 = estimate_cost(
            materialize_workflow(three), three.price_overrides
        )

        # More interrogation rounds ⇒ strictly more calls (interrogation is the
        # dominant call class), but generate/improve/peer-review are unchanged.
        assert est3.total_calls > one.total_calls

    def test_band_is_ordered(self) -> None:
        slim = _flagship_diamond_config()
        workflow = materialize_workflow(slim)
        est = estimate_cost(workflow, slim.price_overrides)
        assert est.total_min <= est.total_expected <= est.total_max
        for c in est.competitors:
            assert c.cost_min <= c.cost_expected <= c.cost_max


class TestPricing:
    def test_ollama_priced_at_zero(self) -> None:
        slim = _ollama_diamond_config()
        workflow = materialize_workflow(slim)
        est = estimate_cost(workflow, slim.price_overrides)
        assert est.total_expected == pytest.approx(0.0, abs=1e-12)
        assert all(
            c.price.input_per_token == pytest.approx(0.0, abs=1e-12)
            for c in est.competitors
        )

    def test_flagships_priced_above_zero(self) -> None:
        slim = _flagship_diamond_config()
        workflow = materialize_workflow(slim)
        est = estimate_cost(workflow, slim.price_overrides)
        assert est.total_expected > 0.0

    def test_unknown_model_fails_loud(self) -> None:
        with pytest.raises(ConfigurationError, match="No price found"):
            resolve_price("no-such-model-xyz-2099", "openai", {})

    def test_price_override_used(self) -> None:
        price = resolve_price(
            "no-such-model-xyz-2099",
            "openai",
            {
                "no-such-model-xyz-2099": {
                    "input_per_1m": 3.0,
                    "output_per_1m": 9.0,
                }
            },
        )
        assert price.input_per_token == pytest.approx(3e-6)
        assert price.output_per_token == pytest.approx(9e-6)
        assert "override" in price.source


class TestReport:
    def test_assumptions_disclose_caching_and_call_regime(self) -> None:
        slim = _flagship_diamond_config()
        workflow = materialize_workflow(slim)
        est = estimate_cost(workflow, slim.price_overrides)
        blob = " ".join(est.assumptions).lower()
        assert "caching" in blob
        assert "no memoization" in blob
        assert "scores fail to parse" in blob

    def test_output_priors_cover_every_stage(self) -> None:
        for stage in (
            "generate",
            "interrogate_questions",
            "interrogate_answers",
            "diverge_improve",
            "peer_review",
            "converge_improve",
        ):
            assert stage in OUTPUT_PRIORS


def test_cost_estimate_from_example_config_file() -> None:
    example = Path("config.example.yml")
    if not example.is_file():
        pytest.skip("config.example.yml not present")
    raw = yaml.safe_load(example.read_text(encoding="utf-8"))
    slim = SlimConfig(**raw)
    workflow = materialize_workflow(slim)
    est = estimate_cost(workflow, slim.price_overrides)
    assert est.total_calls > 0
    assert est.total_expected > 0.0
