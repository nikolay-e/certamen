import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from certamen.domain.disagreement.detector import Disagreement
from certamen.ports.llm import BaseModel

_EVIDENCE_PROMPT = """\
In this discussion about "{question}", you claimed:
"{own_claim}"

The other model claimed:
"{other_claim}"

Present your strongest evidence for your position. What specific facts, \
reasoning, or examples support your claim?
Also: What evidence would change your mind?"""

_NEUTRAL_ANALYSIS_PROMPT = """\
Disagreement on "{question}" — Topic: {topic}

{model_a}: {claim_a}
{model_b}: {claim_b}

Evidence {model_a}: {evidence_a}
Evidence {model_b}: {evidence_b}

Briefly: (1) What evidence supports? (2) Incompatible or different aspects? (3) What would settle it?
Status: resolved / partially_resolved / unresolved. Confidence (0.0-1.0):"""


@dataclass
class DisagreementReport:
    topic: str
    positions: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)
    neutral_analysis: str = ""
    resolution_status: str = "unresolved"
    confidence: float = 0.0


class DisagreementInvestigator:
    async def investigate(
        self,
        disagreement: Disagreement,
        models: Mapping[str, BaseModel],
        question: str,
    ) -> DisagreementReport:
        model_names = list(disagreement.positions.keys())
        claims = list(disagreement.positions.values())

        evidence: dict[str, str] = {}
        for i, model_name in enumerate(model_names):
            model = self._find_model(model_name, models)
            if model is None:
                continue
            other_claim = claims[1 - i] if len(claims) > 1 else ""
            prompt = _EVIDENCE_PROMPT.format(
                question=question,
                own_claim=claims[i],
                other_claim=other_claim,
            )
            response = await model.generate(prompt)
            if not response.is_error():
                evidence[model_name] = response.content.strip()

        neutral_model = self._pick_neutral_model(model_names, models)
        neutral_analysis = ""
        resolution_status = "unresolved"
        confidence = 0.0

        if neutral_model and len(model_names) >= 2:
            prompt = _NEUTRAL_ANALYSIS_PROMPT.format(
                question=question,
                topic=disagreement.topic,
                model_a=model_names[0],
                claim_a=claims[0],
                model_b=model_names[1],
                claim_b=claims[1] if len(claims) > 1 else "",
                evidence_a=evidence.get(
                    model_names[0], "No evidence provided"
                ),
                evidence_b=evidence.get(
                    model_names[1] if len(model_names) > 1 else "",
                    "No evidence provided",
                ),
            )
            response = await neutral_model.generate(prompt)
            if not response.is_error():
                neutral_analysis = response.content.strip()
                resolution_status, confidence = self._parse_resolution(
                    neutral_analysis
                )

        return DisagreementReport(
            topic=disagreement.topic,
            positions=disagreement.positions,
            evidence=evidence,
            neutral_analysis=neutral_analysis,
            resolution_status=resolution_status,
            confidence=confidence,
        )

    @staticmethod
    def _find_model(
        model_name: str, models: Mapping[str, BaseModel]
    ) -> BaseModel | None:
        name_lower = model_name.lower().strip()
        return (
            _find_exact_match(name_lower, models)
            or _find_prefix_match(name_lower, models)
            or _find_substring_match(name_lower, models)
        )

    @staticmethod
    def _pick_neutral_model(
        disputant_names: list[str], models: Mapping[str, BaseModel]
    ) -> BaseModel | None:
        for key, model in models.items():
            if key not in disputant_names and (
                not model.display_name
                or model.display_name not in disputant_names
            ):
                return model
        return next(iter(models.values()), None)

    @staticmethod
    def _parse_resolution(analysis: str) -> tuple[str, float]:
        text = analysis.lower()
        status = "unresolved"
        if re.search(r"partially[\s_]resolved", text):
            status = "partially_resolved"
        elif re.search(r"\bunresolved\b", text):
            status = "unresolved"
        elif re.search(r"\bresolved\b", text):
            status = "resolved"
        conf_match = re.search(
            r"confidence[:\s]+([01]?\.\d+)", text, re.IGNORECASE
        )
        if not conf_match:
            conf_match = re.search(r"\b([01]\.\d+)\b", text)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        return status, min(1.0, max(0.0, confidence))


def _find_exact_match(
    name_lower: str, models: Mapping[str, BaseModel]
) -> BaseModel | None:
    for key, model in models.items():
        if key.lower() == name_lower:
            return model
        if model.display_name and model.display_name.lower() == name_lower:
            return model
    return None


def _find_prefix_match(
    name_lower: str, models: Mapping[str, BaseModel]
) -> BaseModel | None:
    for _, model in models.items():
        if model.display_name and model.display_name.lower().startswith(
            name_lower
        ):
            return model
    return None


def _find_substring_match(
    name_lower: str, models: Mapping[str, BaseModel]
) -> BaseModel | None:
    for key, model in models.items():
        if name_lower in key.lower():
            return model
        if model.display_name and name_lower in model.display_name.lower():
            return model
    return None
