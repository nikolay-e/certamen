from certamen_core.domain.knowledge_map.builder import KnowledgeMap


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

        if km.consensus:
            lines += ["## Consensus (All Models Agree)", ""]
            for item in km.consensus:
                lines.append(f"- {item.claim} [{item.confidence}]")
            lines.append("")

        if km.disagreements:
            lines += ["## Disagreements", ""]
            for d in km.disagreements:
                lines.append(f"### {d.topic}")
                for model, position in d.positions.items():
                    lines.append(f"- **{model}**: {position}")
                if d.neutral_analysis:
                    summary = d.neutral_analysis[:400]
                    lines.append(f"- **Analysis**: {summary}")
                lines.append(
                    f"- **Status**: {d.resolution_status} "
                    f"(confidence: {d.confidence:.0%})"
                )
                lines.append("")

        if km.unique_insights:
            lines += ["## Unique Insights (Single-Source)", ""]
            for model, insights in km.unique_insights.items():
                lines.append(f"### From {model}")
                for insight in insights:
                    lines.append(f"- {insight}")
                lines.append("")

        if km.known_unknowns:
            lines += ["## Known Unknowns", ""]
            for unknown in km.known_unknowns:
                lines.append(f"- {unknown}")
            lines.append("")

        if km.assumptions:
            lines += ["## Assumptions Made", ""]
            for assumption in km.assumptions:
                lines.append(f"- {assumption}")
            lines.append("")

        dist = {k: v for k, v in km.confidence_distribution.items() if v > 0}
        if dist:
            lines += ["## Confidence Distribution", ""]
            for level, count in dist.items():
                lines.append(f"- {level}: {count} claims")
            lines.append("")

        if km.exploration_branches:
            lines += ["## Suggested Follow-Up Questions", ""]
            for i, question in enumerate(km.exploration_branches, 1):
                lines.append(f"{i}. {question}")
            lines.append("")

        return "\n".join(lines)

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
