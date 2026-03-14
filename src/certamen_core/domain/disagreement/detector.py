import re
from dataclasses import dataclass, field

from certamen_core.ports.llm import BaseModel

_DETECTION_PROMPT = """\
Compare these responses and identify specific points of DISAGREEMENT.
Not style differences — substantive factual or analytical disagreements.

Question: {question}

{responses}

For each disagreement, output in this exact format:
DISAGREEMENT: [topic]
POSITION_A: [model name]: [their claim]
POSITION_B: [model name]: [their claim]
SIGNIFICANCE: [why this matters for answering the question]
---

If there are no substantive disagreements, output: NO_DISAGREEMENTS"""


@dataclass
class Disagreement:
    topic: str
    positions: dict[str, str] = field(default_factory=dict)
    significance: str = ""


class DisagreementDetector:
    async def detect_disagreements(
        self,
        responses: dict[str, str],
        judge_model: BaseModel,
        question: str,
    ) -> list[Disagreement]:
        if len(responses) < 2:
            return []

        formatted = "\n\n".join(
            f"=== {name} ===\n{text}" for name, text in responses.items()
        )
        prompt = _DETECTION_PROMPT.format(
            question=question, responses=formatted
        )

        response = await judge_model.generate(prompt)
        if response.is_error():
            return []

        return self._parse_disagreements(response.content)

    @staticmethod
    def _parse_disagreements(text: str) -> list[Disagreement]:
        if "NO_DISAGREEMENTS" in text.upper():
            return []

        disagreements = []
        blocks = re.split(r"\n---+\n?", text)

        for raw_block in blocks:
            block = raw_block.strip()
            if not block:
                continue

            topic_match = re.search(r"DISAGREEMENT:\s*(.+)", block)
            pos_a_match = re.search(
                r"POSITION_A:\s*([^:]+):\s*(.+?)(?=\nPOSITION_B:|\Z)",
                block,
                re.DOTALL,
            )
            pos_b_match = re.search(
                r"POSITION_B:\s*([^:]+):\s*(.+?)(?=\nSIGNIFICANCE:|\Z)",
                block,
                re.DOTALL,
            )
            sig_match = re.search(
                r"SIGNIFICANCE:\s*(.+?)(?=\n---|\Z)", block, re.DOTALL
            )

            if not (topic_match and pos_a_match and pos_b_match):
                continue

            positions = {
                pos_a_match.group(1).strip(): pos_a_match.group(2).strip(),
                pos_b_match.group(1).strip(): pos_b_match.group(2).strip(),
            }

            disagreements.append(
                Disagreement(
                    topic=topic_match.group(1).strip(),
                    positions=positions,
                    significance=(
                        sig_match.group(1).strip() if sig_match else ""
                    ),
                )
            )

        return disagreements
