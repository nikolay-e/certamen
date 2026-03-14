import re
from dataclasses import dataclass


@dataclass
class ConfidenceClaim:
    claim: str
    confidence: str
    context: str


class ConfidenceParser:
    _CONFIDENCE_PATTERN = re.compile(
        r"([^\[\n]{10,200}?)\s*\[(HIGH|MEDIUM|LOW|UNCERTAIN)\]",
        re.IGNORECASE,
    )
    _KNOWN_UNKNOWNS_PATTERN = re.compile(
        r"KNOWN[_\s]UNKNOWNS?:\s*(.+)(?=\n[A-Z_]+:|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    _ASSUMPTIONS_PATTERN = re.compile(
        r"ASSUMPTIONS?:\s*(.+)(?=\n[A-Z_]+:|\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    def parse_confidence_tags(self, response: str) -> list[ConfidenceClaim]:
        claims = []
        for match in self._CONFIDENCE_PATTERN.finditer(response):
            claims.append(
                ConfidenceClaim(
                    claim=match.group(1).strip(),
                    confidence=match.group(2).upper(),
                    context=response[
                        max(0, match.start() - 100) : match.end() + 100
                    ],
                )
            )
        return claims

    def extract_known_unknowns(self, response: str) -> list[str]:
        match = self._KNOWN_UNKNOWNS_PATTERN.search(response)
        if not match:
            return []
        return self._split_list_items(match.group(1))

    def extract_assumptions(self, response: str) -> list[str]:
        match = self._ASSUMPTIONS_PATTERN.search(response)
        if not match:
            return []
        return self._split_list_items(match.group(1))

    @staticmethod
    def _split_list_items(text: str) -> list[str]:
        items = []
        for raw in text.strip().split("\n"):
            cleaned = re.sub(r"^[\s•\-\*\d\.]+", "", raw).strip()
            if cleaned:
                items.append(cleaned)
        return items
