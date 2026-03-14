import asyncio
from typing import Any

from certamen_core.application.workflow.nodes.base import (
    BaseNode,
    ExecutionContext,
    Port,
    PortType,
)
from certamen_core.application.workflow.registry import register_node
from certamen_core.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)


@register_node
class DisagreementDetectionNode(BaseNode):
    NODE_TYPE = "tournament/detect_disagreements"
    DISPLAY_NAME = "Detect Disagreements"
    CATEGORY = "Tournament"
    DESCRIPTION = (
        "Identify substantive disagreements between model responses."
        " A judge model compares responses and extracts points where models"
        " hold genuinely different positions."
    )

    INPUTS = [
        Port(
            "responses",
            PortType.RESPONSES,
            required=True,
            description="Model responses to compare for disagreements",
        ),
        Port(
            "question",
            PortType.STRING,
            required=True,
            description="The original question being analyzed",
        ),
        Port(
            "judge",
            PortType.MODEL,
            required=True,
            description="Model to use as disagreement detector",
        ),
    ]

    OUTPUTS = [
        Port(
            "disagreements",
            PortType.RESULTS,
            description="Detected disagreement objects",
        ),
        Port(
            "disagreement_report",
            PortType.STRING,
            description="Human-readable summary of disagreements",
        ),
    ]

    PROPERTIES = {}

    async def execute(
        self, inputs: dict[str, Any], context: ExecutionContext
    ) -> dict[str, Any]:
        from certamen_core.domain.disagreement.detector import (
            DisagreementDetector,
        )
        from certamen_core.infrastructure.llm.factory import (
            ensure_single_model_instance,
        )

        responses: dict[str, str] = inputs.get("responses") or {}
        question: str = inputs.get("question") or context.question
        judge_input = inputs.get("judge")

        if not responses or not question or not judge_input:
            return {"disagreements": [], "disagreement_report": ""}

        judge = await ensure_single_model_instance(judge_input, "judge")
        if not judge:
            return {"disagreements": [], "disagreement_report": ""}

        detector = DisagreementDetector()
        disagreements = await detector.detect_disagreements(
            responses, judge, question
        )

        if not disagreements:
            return {
                "disagreements": [],
                "disagreement_report": "No substantive disagreements detected.",
            }

        report_lines = [f"Found {len(disagreements)} disagreement(s):\n"]
        for i, d in enumerate(disagreements, 1):
            report_lines.append(f"{i}. **{d.topic}** — {d.significance}")
            for model, pos in d.positions.items():
                report_lines.append(f"   - {model}: {pos[:200]}")

        return {
            "disagreements": disagreements,
            "disagreement_report": "\n".join(report_lines),
        }


@register_node
class DisagreementInvestigationNode(BaseNode):
    NODE_TYPE = "tournament/investigate_disagreement"
    DISPLAY_NAME = "Investigate Disagreements"
    CATEGORY = "Tournament"
    DESCRIPTION = (
        "Deep-dive into detected disagreements. Each disagreeing model presents"
        " evidence for its position; a neutral model analyzes the evidence."
    )

    INPUTS = [
        Port(
            "disagreements",
            PortType.RESULTS,
            required=True,
            description="Disagreement objects from detect_disagreements node",
        ),
        Port(
            "models",
            PortType.MODELS,
            required=True,
            description="All available models (for evidence gathering and neutral analysis)",
        ),
        Port(
            "question",
            PortType.STRING,
            required=True,
            description="The original question",
        ),
    ]

    OUTPUTS = [
        Port(
            "investigation_report",
            PortType.STRING,
            description="Full investigation report in markdown",
        ),
        Port(
            "new_knowledge",
            PortType.INSIGHTS,
            description="Knowledge extracted from disagreement investigation",
        ),
    ]

    PROPERTIES = {}

    async def execute(
        self, inputs: dict[str, Any], context: ExecutionContext
    ) -> dict[str, Any]:
        from certamen_core.domain.disagreement.resolver import (
            DisagreementInvestigator,
        )

        disagreements = inputs.get("disagreements") or []
        question: str = inputs.get("question") or context.question
        models_input = inputs.get("models")

        if not disagreements or not question:
            return {"investigation_report": "", "new_knowledge": []}

        models_result, empty = await self.ensure_models_or_empty(models_input)
        if empty is not None:
            return {"investigation_report": "", "new_knowledge": []}
        models = models_result

        investigator = DisagreementInvestigator()

        reports = await asyncio.gather(
            *[
                investigator.investigate(d, models, question)
                for d in disagreements
            ],
            return_exceptions=True,
        )

        from certamen_core.domain.disagreement.resolver import (
            DisagreementReport,
        )

        valid_reports: list[DisagreementReport] = [
            r for r in reports if isinstance(r, DisagreementReport)  # type: ignore[misc]
        ]
        failed = sum(1 for r in reports if isinstance(r, BaseException))
        if failed:
            logger.warning(
                "DisagreementInvestigation: %d pair(s) failed", failed
            )

        new_knowledge: list[str] = []
        md_lines: list[str] = ["# Disagreement Investigation Report\n"]

        for report in valid_reports:
            md_lines.append(f"## {report.topic}\n")
            for model, pos in report.positions.items():
                md_lines.append(f"**{model}**: {pos}\n")
            if report.evidence:
                md_lines.append("### Evidence\n")
                for model, ev in report.evidence.items():
                    md_lines.append(f"**{model}**: {ev[:400]}\n")
                    new_knowledge.append(f"[{model}] {ev[:300]}")
            if report.neutral_analysis:
                md_lines.append(
                    f"### Analysis\n{report.neutral_analysis[:600]}\n"
                )
            md_lines.append(
                f"**Status**: {report.resolution_status} "
                f"(confidence: {report.confidence:.0%})\n"
            )
            md_lines.append("---\n")

        return {
            "investigation_report": "\n".join(md_lines),
            "new_knowledge": new_knowledge,
        }
