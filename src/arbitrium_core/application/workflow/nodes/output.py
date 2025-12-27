from typing import Any

from arbitrium_core.application.workflow.nodes.base import (
    BaseNode,
    ExecutionContext,
    Port,
    PortType,
)
from arbitrium_core.application.workflow.registry import register_node


@register_node
class ChampionNode(BaseNode):
    NODE_TYPE = "tournament/champion"
    DISPLAY_NAME = "Champion"
    CATEGORY = "Tournament"
    DESCRIPTION = "Display tournament champion and final response"
    INPUTS = [
        Port(
            "model",
            PortType.MODEL,
            description="Connect Rank node's champion output here to display the winner",
        ),
        Port(
            "response",
            PortType.STRING,
            required=False,
            description="Winner's final answer text (from Generate or Improve node)",
        ),
        Port(
            "rankings",
            PortType.RANKINGS,
            required=False,
            description="Full leaderboard from Rank node showing all model placements",
        ),
    ]
    OUTPUTS = []
    PROPERTIES = {}

    async def execute(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        model = inputs.get("model")
        response = inputs.get("response", "")
        rankings = inputs.get("rankings", [])

        result = {
            "champion": {
                "model": model,
                "response": response,
                "rankings": rankings,
            }
        }

        await self.broadcast_result("champion_declared", result, context)

        return result


@register_node
class RankingsNode(BaseNode):
    NODE_TYPE = "tournament/rankings"
    DISPLAY_NAME = "Rankings"
    CATEGORY = "Tournament"
    DESCRIPTION = "Display final tournament rankings"
    INPUTS = [
        Port(
            "rankings",
            PortType.RANKINGS,
            description="Connect Rank node's rankings output to display the leaderboard",
        ),
        Port(
            "scores",
            PortType.SCORES,
            required=False,
            description="Optional: connect scores from evaluation node to show alongside rankings",
        ),
    ]
    OUTPUTS = []
    PROPERTIES = {
        "format": {
            "type": "string",
            "default": "table",
            "enum": ["table", "list", "json"],
            "description": "How to display results: table (columns), list (bullet points), json (raw data)",
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        rankings = inputs.get("rankings", [])
        scores = inputs.get("scores", {})

        result = {
            "rankings": rankings,
            "scores": scores,
        }

        await self.broadcast_result("rankings_output", result, context)

        return result


@register_node
class ReportNode(BaseNode):
    NODE_TYPE = "tournament/report"
    DISPLAY_NAME = "Report"
    CATEGORY = "Tournament"
    DESCRIPTION = "Generate tournament report in various formats"
    INPUTS = [
        Port(
            "champion",
            PortType.MODEL,
            required=False,
            description="Connect Rank node's champion output to show winner in report",
        ),
        Port(
            "rankings",
            PortType.RANKINGS,
            required=False,
            description="Connect Rank node's rankings output to include leaderboard",
        ),
        Port(
            "eliminated_info",
            PortType.RESULTS,
            required=False,
            description="Connect Eliminate node's eliminated_info to show elimination history",
        ),
        Port(
            "question",
            PortType.STRING,
            required=False,
            description="Connect original prompt Text node to include the question in report",
        ),
    ]
    OUTPUTS = [
        Port(
            "report",
            PortType.STRING,
            description="Complete tournament summary - can be saved or displayed",
        ),
    ]
    PROPERTIES = {
        "format": {
            "type": "string",
            "default": "markdown",
            "enum": ["markdown", "json", "text"],
            "description": "markdown: headers and formatting, json: machine-readable, text: plain",
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        champion = inputs.get("champion")
        rankings = inputs.get("rankings", [])
        eliminated_info = inputs.get("eliminated_info", [])
        question = inputs.get("question", "")
        fmt = self.node_properties.get("format", "markdown")

        if fmt == "markdown":
            lines = ["# Tournament Report\n"]

            if question:
                lines.append(f"## Question\n{question}\n")

            if champion:
                model_name = getattr(champion, "display_name", str(champion))
                lines.append(f"## Champion\n**{model_name}**\n")

            if rankings:
                lines.append("## Final Rankings\n")
                for r in rankings:
                    lines.append(
                        f"{r['rank']}. {r['model']} (score: {r.get('score', 'N/A')})"
                    )
                lines.append("")

            if eliminated_info:
                lines.append("## Elimination History\n")
                for e in eliminated_info:
                    lines.append(
                        f"- Round {e.get('round', '?')}: {e['model']} (score: {e.get('score', 'N/A')})"
                    )

            report = "\n".join(lines)
        elif fmt == "json":
            import json

            report = json.dumps(
                {
                    "question": question,
                    "champion": str(champion) if champion else None,
                    "rankings": rankings,
                    "eliminated": eliminated_info,
                },
                indent=2,
            )
        else:
            lines = ["Tournament Report"]
            if champion:
                lines.append(f"Champion: {champion}")
            if rankings:
                lines.append("Rankings:")
                for r in rankings:
                    lines.append(f"  {r['rank']}. {r['model']}")
            report = "\n".join(lines)

        return {"report": report}
