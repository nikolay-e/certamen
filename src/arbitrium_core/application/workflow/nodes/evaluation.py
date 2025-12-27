import asyncio
from typing import Any

from arbitrium_core.application.workflow.nodes.base import (
    BaseNode,
    ExecutionContext,
    Port,
    PortType,
    build_evaluation_prompt,
    require_inputs,
    safe_generate,
)
from arbitrium_core.application.workflow.registry import register_node
from arbitrium_core.domain.tournament.anonymizer import ModelAnonymizer
from arbitrium_core.domain.tournament.scoring import ScoreExtractor
from arbitrium_core.shared.statistics import aggregate_scores

_scorer = ScoreExtractor()
_anonymizer = ModelAnonymizer()


@register_node
class PeerReviewNode(BaseNode):
    NODE_TYPE = "tournament/peer_review"
    DISPLAY_NAME = "Peer Review"
    CATEGORY = "Tournament"
    DESCRIPTION = "Models evaluate each other's responses"
    INPUTS = [
        Port(
            "models",
            PortType.MODELS,
            description="Models that will act as judges, scoring each other's work",
        ),
        Port(
            "responses",
            PortType.RESPONSES,
            description="The generated answers to be judged (from Generate node)",
        ),
        Port(
            "question",
            PortType.STRING,
            description="The original prompt - judges need this to evaluate answer quality",
        ),
    ]
    OUTPUTS = [
        Port(
            "scores",
            PortType.SCORES,
            description="Final score for each model (1-10), averaged across all peer reviews",
        ),
        Port(
            "evaluations",
            PortType.RESPONSES,
            description="Full evaluation text explaining why each judge gave their scores",
        ),
    ]
    PROPERTIES = {
        "anonymize": {
            "type": "boolean",
            "default": True,
            "description": "Show responses as 'Model A/B/C' instead of real names to prevent favoritism",
        },
        "criteria": {
            "type": "string",
            "default": "",
            "multiline": True,
            "description": "What to evaluate (e.g., 'accuracy, code quality, explanation clarity'). Leave empty for general quality",
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
        question = inputs.get("question", "")
        anonymize = self.node_properties.get("anonymize", True)
        criteria = self.node_properties.get("criteria", "")

        if anonymize:
            eval_responses, reverse_mapping = _anonymizer.anonymize_responses(
                responses
            )
            eval_names = list(eval_responses.keys())
        else:
            eval_responses = responses
            eval_names = list(responses.keys())
            reverse_mapping = {}

        original_names = list(responses.keys())
        all_scores: dict[str, list[float]] = {
            name: [] for name in original_names
        }
        evaluations: dict[str, str] = {}

        prompt = build_evaluation_prompt(question, eval_responses, criteria)

        async def evaluate_one(
            evaluator_key: str, evaluator: Any
        ) -> tuple[str, str, dict[str, float]]:
            text, success = await safe_generate(evaluator, prompt)
            if not success:
                return evaluator_key, "", {}
            scores = _scorer.extract_scores_from_evaluation(
                text, eval_names, evaluator_key
            )
            return evaluator_key, text, scores

        tasks = [evaluate_one(k, m) for k, m in models.items()]
        results = await asyncio.gather(*tasks)

        for evaluator_key, text, scores in results:
            if text:
                evaluations[evaluator_key] = text
            for name, score in scores.items():
                original_name = reverse_mapping.get(name, name)
                if original_name in all_scores:
                    all_scores[original_name].append(score)

        final_scores = aggregate_scores(all_scores, method="mean")

        return {"scores": final_scores, "evaluations": evaluations}


@register_node
class JudgeEvalNode(BaseNode):
    NODE_TYPE = "tournament/judge"
    DISPLAY_NAME = "Judge"
    CATEGORY = "Tournament"
    DESCRIPTION = "Single judge model evaluates all responses"
    INPUTS = [
        Port(
            "judge",
            PortType.MODEL,
            description="The LLM that will evaluate all responses (connect an LLM node's model_config)",
        ),
        Port(
            "responses",
            PortType.RESPONSES,
            description="The generated answers to be judged (from Generate node)",
        ),
        Port(
            "question",
            PortType.STRING,
            description="The original prompt - judge needs this to evaluate answer quality",
        ),
    ]
    OUTPUTS = [
        Port(
            "scores",
            PortType.SCORES,
            description="Score (1-10) for each model based on judge's evaluation",
        ),
        Port(
            "evaluation",
            PortType.STRING,
            description="Judge's full reasoning explaining the scores given",
        ),
    ]
    PROPERTIES = {
        "criteria": {
            "type": "string",
            "default": "",
            "multiline": True,
            "description": "What to evaluate (e.g., 'correctness, reasoning, completeness'). Leave empty for general quality",
        },
    }

    @require_inputs("judge", "responses")
    async def execute(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        models_dict, error = await self.ensure_models_or_empty(
            inputs["judge"], single=True
        )
        if error:
            return error
        judge = models_dict["model"]

        responses = inputs["responses"]
        question = inputs.get("question", "")

        criteria = self.node_properties.get("criteria", "")
        model_names = list(responses.keys())
        prompt = build_evaluation_prompt(question, responses, criteria)

        text, success = await safe_generate(judge, prompt)
        if not success:
            return {"scores": {}, "evaluation": ""}

        scores = _scorer.extract_scores_from_evaluation(
            text, model_names, "judge"
        )
        return {"scores": scores, "evaluation": text}
