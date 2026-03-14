import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from certamen_core.domain.confidence.parser import ConfidenceParser

if TYPE_CHECKING:
    from certamen_core.domain.disagreement.resolver import DisagreementReport
    from certamen_core.ports.llm import BaseModel

_CONSENSUS_PROMPT = """\
Analyze these responses and identify claims that ALL or MOST models agree on.

Question: {question}

{responses}

List only genuine consensus points — claims explicitly or implicitly shared across responses.
Format each as:
CONSENSUS: [claim] [HIGH|MEDIUM|LOW confidence in consensus]"""

_UNIQUE_INSIGHTS_PROMPT = """\
Identify insights that appear in ONLY ONE response — things others did not mention.

Question: {question}

{responses}

For each unique insight, output:
MODEL: [model name]
INSIGHT: [the unique insight not found in other responses]
---

Only list genuinely unique insights, not just different phrasings of the same idea."""

_NONE_IDENTIFIED = "None identified"

_EXPLORATION_PROMPT = """\
Given this knowledge map about "{question}", identify 5-8 follow-up questions \
that would extract the MOST additional knowledge.

=== CONSENSUS (established facts) ===
{consensus}

=== UNRESOLVED DISAGREEMENTS (highest-value targets) ===
{disagreements}

=== KNOWN UNKNOWNS (gaps models identified) ===
{known_unknowns}

=== LOW-CONFIDENCE CLAIMS (need verification) ===
{low_confidence}

=== ASSUMPTIONS (untested premises) ===
{assumptions}

For each question:
1. State the question
2. Explain WHY answering it would yield high-value knowledge
3. Indicate which category above it targets

Prioritize questions that would:
- Settle unresolved disagreements with evidence
- Fill the most critical known unknowns
- Test the riskiest assumptions
- Verify low-confidence claims that, if true, would be highly significant

Output numbered questions, one per line."""


@dataclass
class ConsensusItem:
    claim: str
    confidence: str


@dataclass
class KnowledgeMap:
    question: str
    consensus: list[ConsensusItem] = field(default_factory=list)
    disagreements: list[Any] = field(default_factory=list)
    unique_insights: dict[str, list[str]] = field(default_factory=dict)
    known_unknowns: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    confidence_distribution: dict[str, int] = field(default_factory=dict)
    synthesis: str = ""
    champion_model: str = ""
    exploration_branches: list[str] = field(default_factory=list)


class KnowledgeMapBuilder:
    def __init__(self) -> None:
        self._confidence_parser = ConfidenceParser()

    async def build(
        self,
        question: str,
        all_responses: dict[str, str],
        synthesis: str,
        champion_model: str,
        judge_model: "BaseModel",
        disagreements: list["DisagreementReport"] | None = None,
    ) -> KnowledgeMap:
        consensus = await self._detect_consensus(
            question, all_responses, judge_model
        )
        unique_insights = await self._extract_unique_insights(
            question, all_responses, judge_model
        )

        known_unknowns: list[str] = []
        assumptions: list[str] = []
        confidence_dist: dict[str, int] = {
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
            "UNCERTAIN": 0,
        }

        for response_text in all_responses.values():
            known_unknowns.extend(
                self._confidence_parser.extract_known_unknowns(response_text)
            )
            assumptions.extend(
                self._confidence_parser.extract_assumptions(response_text)
            )
            for claim in self._confidence_parser.parse_confidence_tags(
                response_text
            ):
                confidence_dist[claim.confidence] = (
                    confidence_dist.get(claim.confidence, 0) + 1
                )

        return KnowledgeMap(
            question=question,
            consensus=consensus,
            disagreements=disagreements or [],
            unique_insights=unique_insights,
            known_unknowns=list(dict.fromkeys(known_unknowns)),
            assumptions=list(dict.fromkeys(assumptions)),
            confidence_distribution=confidence_dist,
            synthesis=synthesis,
            champion_model=champion_model,
            exploration_branches=[],
        )

    async def generate_exploration_branches(
        self,
        km: KnowledgeMap,
        model: "BaseModel",
    ) -> list[str]:
        consensus_text = (
            "\n".join(
                f"- {c.claim} [{c.confidence}]" for c in km.consensus[:10]
            )
            or _NONE_IDENTIFIED
        )

        disagreement_lines = []
        for d in km.disagreements[:5]:
            positions_str = "; ".join(
                f"{m}: {p[:100]}" for m, p in d.positions.items()
            )
            status = getattr(d, "resolution_status", "unresolved")
            disagreement_lines.append(
                f"- {d.topic} ({status}): {positions_str}"
            )
        disagreements_text = "\n".join(disagreement_lines) or _NONE_IDENTIFIED

        unknowns_text = (
            "\n".join(f"- {u}" for u in km.known_unknowns[:10])
            or _NONE_IDENTIFIED
        )

        low_conf = km.confidence_distribution.get(
            "LOW", 0
        ) + km.confidence_distribution.get("UNCERTAIN", 0)
        low_conf_text = (
            f"{km.confidence_distribution.get('LOW', 0)} LOW-confidence claims, "
            f"{km.confidence_distribution.get('UNCERTAIN', 0)} UNCERTAIN claims detected"
            if low_conf > 0
            else _NONE_IDENTIFIED
        )

        assumptions_text = (
            "\n".join(f"- {a}" for a in km.assumptions[:10])
            or _NONE_IDENTIFIED
        )

        prompt = _EXPLORATION_PROMPT.format(
            question=km.question,
            consensus=consensus_text,
            disagreements=disagreements_text,
            known_unknowns=unknowns_text,
            low_confidence=low_conf_text,
            assumptions=assumptions_text,
        )
        response = await model.generate(prompt)
        if response.is_error():
            return []
        return self._parse_questions(response.content)

    async def _detect_consensus(
        self,
        question: str,
        responses: dict[str, str],
        judge_model: "BaseModel",
    ) -> list[ConsensusItem]:
        formatted = "\n\n".join(
            f"=== {name} ===\n{text}" for name, text in responses.items()
        )
        prompt = _CONSENSUS_PROMPT.format(
            question=question, responses=formatted
        )
        response = await judge_model.generate(prompt)
        if response.is_error():
            return []
        return self._parse_consensus(response.content)

    async def _extract_unique_insights(
        self,
        question: str,
        responses: dict[str, str],
        judge_model: "BaseModel",
    ) -> dict[str, list[str]]:
        if len(responses) < 2:
            return {}
        formatted = "\n\n".join(
            f"=== {name} ===\n{text}" for name, text in responses.items()
        )
        prompt = _UNIQUE_INSIGHTS_PROMPT.format(
            question=question, responses=formatted
        )
        response = await judge_model.generate(prompt)
        if response.is_error():
            return {}
        return self._parse_unique_insights(response.content)

    @staticmethod
    def _parse_consensus(text: str) -> list[ConsensusItem]:
        items = []
        for match in re.finditer(
            r"CONSENSUS:\s*(.+?)\s*\[(HIGH|MEDIUM|LOW)\]", text, re.IGNORECASE
        ):
            items.append(
                ConsensusItem(
                    claim=match.group(1).strip(),
                    confidence=match.group(2).upper(),
                )
            )
        return items

    @staticmethod
    def _parse_unique_insights(text: str) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for raw_block in re.split(r"\n---+\n?", text):
            stripped_block = raw_block.strip()
            model_match = re.search(r"MODEL:\s*(.+)", stripped_block)
            insight_match = re.search(
                r"INSIGHT:\s*(.+)(?=\n---|\Z)", stripped_block, re.DOTALL
            )
            if model_match and insight_match:
                model_name = model_match.group(1).strip()
                insight = insight_match.group(1).strip()
                result.setdefault(model_name, []).append(insight)
        return result

    @staticmethod
    def _parse_questions(text: str) -> list[str]:
        questions = []
        for raw in text.strip().split("\n"):
            cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", raw).strip()
            if cleaned and "?" in cleaned:
                questions.append(cleaned)
        return questions
