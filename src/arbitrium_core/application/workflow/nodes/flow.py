import random
from collections.abc import Callable
from typing import Any

from arbitrium_core.application.workflow.nodes.base import (
    BaseNode,
    ExecutionContext,
    Port,
    PortType,
    merge_indexed_dicts,
    rank_by_scores,
    split_by_rank,
)
from arbitrium_core.application.workflow.registry import register_node
from arbitrium_core.shared.constants import MAX_MULTI_INPUTS


@register_node
class SplitNode(BaseNode):
    NODE_TYPE = "tournament/split"
    DISPLAY_NAME = "Split"
    CATEGORY = "Tournament"
    DESCRIPTION = "Split models into groups for bracket tournaments"
    INPUTS = [
        Port(
            "models",
            PortType.MODELS,
            description="Collection of models to divide into groups",
        ),
    ]
    OUTPUTS = [
        Port("group_1", PortType.MODELS, description="First subset of models"),
        Port(
            "group_2", PortType.MODELS, description="Second subset of models"
        ),
        Port(
            "group_3",
            PortType.MODELS,
            required=False,
            description="Third subset (if num_groups >= 3)",
        ),
        Port(
            "group_4",
            PortType.MODELS,
            required=False,
            description="Fourth subset (if num_groups = 4)",
        ),
    ]
    PROPERTIES = {
        "num_groups": {
            "type": "integer",
            "default": 2,
            "min": 2,
            "max": 4,
            "description": "How many groups to create (2 for head-to-head, 4 for bracket tournament)",
        },
        "method": {
            "type": "select",
            "default": "random",
            "options": ["random", "sequential", "seeded"],
            "description": "random: shuffled assignment, sequential: in order, seeded: alphabetical for reproducibility",
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        models = inputs.get("models", {})
        num_groups = self.node_properties.get("num_groups", 2)
        method = self.node_properties.get("method", "random")

        if not models:
            return {f"group_{i + 1}": {} for i in range(MAX_MULTI_INPUTS)}

        model_keys = list(models.keys())

        if method == "random":
            random.shuffle(model_keys)
        elif method == "seeded":
            model_keys = sorted(model_keys)

        groups: list[dict[str, Any]] = [{} for _ in range(num_groups)]
        for i, key in enumerate(model_keys):
            groups[i % num_groups][key] = models[key]

        return {
            f"group_{i + 1}": groups[i] if i < len(groups) else {}
            for i in range(MAX_MULTI_INPUTS)
        }


@register_node
class MergeNode(BaseNode):
    NODE_TYPE = "tournament/merge"
    DISPLAY_NAME = "Merge"
    CATEGORY = "Tournament"
    DESCRIPTION = "Combine model groups into single collection"
    INPUTS = [
        Port(
            "input_1",
            PortType.MODELS,
            description="First group of models to combine",
        ),
        Port(
            "input_2",
            PortType.MODELS,
            description="Second group of models to combine",
        ),
        Port(
            "input_3",
            PortType.MODELS,
            required=False,
            description="Optional third group to combine",
        ),
        Port(
            "input_4",
            PortType.MODELS,
            required=False,
            description="Optional fourth group to combine",
        ),
    ]
    OUTPUTS = [
        Port(
            "combined",
            PortType.MODELS,
            description="All input models merged into single collection",
        ),
    ]
    PROPERTIES = {}

    async def execute(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        combined = merge_indexed_dicts(inputs, "input")
        return {"combined": combined}


@register_node
class FilterNode(BaseNode):
    NODE_TYPE = "tournament/filter"
    DISPLAY_NAME = "Filter"
    CATEGORY = "Tournament"
    DESCRIPTION = "Filter models by score threshold or ranking"
    INPUTS = [
        Port(
            "models",
            PortType.MODELS,
            description="Models to filter based on their scores",
        ),
        Port(
            "scores",
            PortType.SCORES,
            description="Score values used to determine pass/fail",
        ),
    ]
    OUTPUTS = [
        Port(
            "passed",
            PortType.MODELS,
            description="Models that met the filter criteria",
        ),
        Port(
            "failed",
            PortType.MODELS,
            description="Models that did not meet the criteria",
        ),
    ]
    PROPERTIES = {
        "threshold": {
            "type": "number",
            "default": 5.0,
            "min": 1.0,
            "max": 10.0,
            "description": "Minimum score to pass (for 'above' mode) or maximum to pass (for 'below' mode)",
        },
        "mode": {
            "type": "select",
            "default": "above",
            "options": ["above", "below", "top_n", "bottom_n"],
            "description": "above/below: use threshold, top_n: keep best N models, bottom_n: keep worst N models",
        },
        "n": {
            "type": "integer",
            "default": 2,
            "min": 1,
            "description": "How many models to keep when using top_n or bottom_n mode",
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        models = inputs.get("models", {})
        scores = inputs.get("scores", {})

        if not models:
            return {"passed": {}, "failed": {}}

        threshold = self.node_properties.get("threshold", 5.0)
        mode = self.node_properties.get("mode", "above")
        n = self.node_properties.get("n", 2)

        if mode in ("above", "below"):
            check: Callable[[float], bool] = (
                (lambda s: s >= threshold)
                if mode == "above"
                else (lambda s: s < threshold)
            )
            passed = {
                k: v for k, v in models.items() if check(scores.get(k, 0))
            }
            failed = {
                k: v for k, v in models.items() if not check(scores.get(k, 0))
            }
        else:
            reverse = mode == "top_n"
            ranked = rank_by_scores(models, scores, reverse=reverse)
            passed, failed = split_by_rank(models, ranked, n)

        return {"passed": passed, "failed": failed}


@register_node
class EliminateNode(BaseNode):
    NODE_TYPE = "tournament/eliminate"
    DISPLAY_NAME = "Eliminate"
    CATEGORY = "Tournament"
    DESCRIPTION = "Eliminate lowest scoring models from tournament"
    INPUTS = [
        Port(
            "models",
            PortType.MODELS,
            description="Connect Models or previous Eliminate node's survivors",
        ),
        Port(
            "scores",
            PortType.SCORES,
            description="Connect Peer Review or Judge node's scores output",
        ),
    ]
    OUTPUTS = [
        Port(
            "survivors",
            PortType.MODELS,
            description="Winners who continue - connect to next round's Generate",
        ),
        Port(
            "eliminated",
            PortType.MODELS,
            description="Losers removed - connect to Extract Insights to save their ideas",
        ),
        Port(
            "eliminated_info",
            PortType.RESULTS,
            description="Elimination record - connect to Report node for history",
        ),
    ]
    PROPERTIES = {
        "count": {
            "type": "integer",
            "default": 1,
            "min": 1,
            "description": "How many worst-scoring models to remove each round (1 = slow elimination, more = faster)",
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        models = inputs.get("models", {})
        scores = inputs.get("scores", {})
        count = self.node_properties.get("count", 1)

        if not models:
            return {"survivors": {}, "eliminated": {}, "eliminated_info": []}

        ranked = rank_by_scores(models, scores, reverse=False)
        eliminated, survivors = split_by_rank(models, ranked, count)

        eliminated_info = [
            {"model": k, "score": scores.get(k, 0), "round": context.round_num}
            for k in ranked[:count]
        ]

        return {
            "survivors": survivors,
            "eliminated": eliminated,
            "eliminated_info": eliminated_info,
        }


@register_node
class GateNode(BaseNode):
    NODE_TYPE = "flow/gate"
    DISPLAY_NAME = "Gate"
    CATEGORY = "Flow"
    DESCRIPTION = "Select between primary input and feedback loop. Outputs champion when only one model remains."
    INPUTS = [
        Port(
            "primary",
            PortType.MODELS,
            description="Initial models (used on first iteration when feedback is empty)",
        ),
        Port(
            "feedback",
            PortType.MODELS,
            required=False,
            description="Survivors from previous round (feedback edge)",
        ),
    ]
    OUTPUTS = [
        Port(
            "models",
            PortType.MODELS,
            description="Models for this round (primary if feedback empty, else feedback)",
        ),
        Port(
            "champion",
            PortType.MODEL,
            required=False,
            description="The winner (output only when exactly 1 model remains)",
        ),
        Port(
            "done",
            PortType.BOOLEAN,
            required=False,
            description="True when tournament is complete (1 model remaining)",
        ),
        Port(
            "round",
            PortType.INTEGER,
            required=False,
            description="Current round number (1-based)",
        ),
    ]
    PROPERTIES = {
        "max_rounds": {
            "type": "integer",
            "default": 10,
            "min": 1,
            "max": 100,
            "description": "Maximum number of rounds to prevent infinite loops",
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        primary = inputs.get("primary", {})
        feedback = inputs.get("feedback")
        max_rounds = self.node_properties.get("max_rounds", 10)

        # Use context.round_num for round tracking instead of instance state
        # This prevents state leakage between executions
        current_round = context.round_num

        if feedback is None or len(feedback) == 0:
            models = primary
        else:
            models = feedback

        model_count = len(models) if models else 0

        if model_count <= 1:
            champion = (
                next(iter(models.values())) if model_count == 1 else None
            )
            return {
                "models": {},
                "champion": champion,
                "done": True,
                "round": current_round,
            }

        if current_round >= max_rounds:
            first_model = next(iter(models.values())) if models else None
            return {
                "models": {},
                "champion": first_model,
                "done": True,
                "round": current_round,
            }

        return {
            "models": models,
            "champion": None,
            "done": False,
            "round": current_round,
        }


@register_node
class RankNode(BaseNode):
    NODE_TYPE = "tournament/rank"
    DISPLAY_NAME = "Rank"
    CATEGORY = "Tournament"
    DESCRIPTION = "Rank models by scores and select champion"
    INPUTS = [
        Port(
            "models",
            PortType.MODELS,
            description="Connect final survivors from Eliminate node or all models",
        ),
        Port(
            "scores",
            PortType.SCORES,
            description="Connect final scores from Peer Review or Judge node",
        ),
    ]
    OUTPUTS = [
        Port(
            "rankings",
            PortType.RANKINGS,
            description="Full leaderboard - connect to Rankings or Report node",
        ),
        Port(
            "champion",
            PortType.MODEL,
            description="The winner (#1 ranked) - connect to Champion node",
        ),
    ]
    PROPERTIES = {}

    async def execute(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        models = inputs.get("models", {})
        scores = inputs.get("scores", {})

        if not models:
            return {"rankings": [], "champion": None}

        ranked = rank_by_scores(models, scores, reverse=True)

        rankings = [
            {"rank": i + 1, "model": k, "score": scores.get(k, 0)}
            for i, k in enumerate(ranked)
        ]

        champion = models.get(ranked[0]) if ranked else None
        return {"rankings": rankings, "champion": champion}
