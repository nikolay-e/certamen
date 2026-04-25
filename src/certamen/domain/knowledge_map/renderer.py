from certamen.domain.knowledge_map.builder import KnowledgeMap


class KnowledgeMapRenderer:
    def to_markdown(self, km: KnowledgeMap) -> str:
        lines: list[str] = [
            f"# Knowledge Map: {km.question}",
            "",
            "## Synthesis (Best Combined Answer)",
            "",
            km.synthesis,
            "",
        ]
        self._append_consensus(lines, km)
        self._append_disagreements(lines, km)
        self._append_unique_insights(lines, km)
        self._append_simple_list(lines, km.known_unknowns, "Known Unknowns")
        self._append_simple_list(lines, km.assumptions, "Assumptions Made")
        self._append_confidence_distribution(lines, km)
        self._append_exploration_branches(lines, km)
        return "\n".join(lines)

    @staticmethod
    def _append_consensus(lines: list[str], km: KnowledgeMap) -> None:
        if not km.consensus:
            return
        lines += ["## Consensus (All Models Agree)", ""]
        for item in km.consensus:
            lines.append(f"- {item.claim} [{item.confidence}]")
        lines.append("")

    @staticmethod
    def _append_disagreements(lines: list[str], km: KnowledgeMap) -> None:
        if not km.disagreements:
            return
        lines += ["## Disagreements", ""]
        for d in km.disagreements:
            lines.append(f"### {d.topic}")
            for model, position in d.positions.items():
                lines.append(f"- **{model}**: {position}")
            if d.neutral_analysis:
                lines.append(f"- **Analysis**: {d.neutral_analysis[:400]}")
            lines.append(
                f"- **Status**: {d.resolution_status} "
                f"(confidence: {d.confidence:.0%})"
            )
            lines.append("")

    @staticmethod
    def _append_unique_insights(lines: list[str], km: KnowledgeMap) -> None:
        if not km.unique_insights:
            return
        lines += ["## Unique Insights (Single-Source)", ""]
        for model, insights in km.unique_insights.items():
            lines.append(f"### From {model}")
            for insight in insights:
                lines.append(f"- {insight}")
            lines.append("")

    @staticmethod
    def _append_simple_list(
        lines: list[str], items: list[str], heading: str
    ) -> None:
        if not items:
            return
        lines += [f"## {heading}", ""]
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    @staticmethod
    def _append_confidence_distribution(
        lines: list[str], km: KnowledgeMap
    ) -> None:
        dist = {k: v for k, v in km.confidence_distribution.items() if v > 0}
        if not dist:
            return
        lines += ["## Confidence Distribution", ""]
        for level, count in dist.items():
            lines.append(f"- {level}: {count} claims")
        lines.append("")

    @staticmethod
    def _append_exploration_branches(
        lines: list[str], km: KnowledgeMap
    ) -> None:
        if not km.exploration_branches:
            return
        lines += ["## Suggested Follow-Up Questions", ""]
        for i, question in enumerate(km.exploration_branches, 1):
            lines.append(f"{i}. {question}")
        lines.append("")

    def to_json(self, km: KnowledgeMap) -> dict[str, object]:
        return {
            "question": km.question,
            "champion_model": km.champion_model,
            "synthesis": km.synthesis,
            "consensus": [
                {"claim": c.claim, "confidence": c.confidence}
                for c in km.consensus
            ],
            "disagreements": [
                {
                    "topic": d.topic,
                    "positions": d.positions,
                    "evidence": d.evidence,
                    "neutral_analysis": d.neutral_analysis,
                    "resolution_status": d.resolution_status,
                    "confidence": d.confidence,
                }
                for d in km.disagreements
            ],
            "unique_insights": km.unique_insights,
            "known_unknowns": km.known_unknowns,
            "assumptions": km.assumptions,
            "confidence_distribution": km.confidence_distribution,
            "exploration_branches": km.exploration_branches,
        }
