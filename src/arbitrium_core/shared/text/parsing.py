def parse_insight_lines(
    text: str,
    min_length: int = 10,
    skip_apologies: bool = True,
) -> list[str]:
    insights = []
    apology_prefixes = ("i cannot", "i'm sorry", "i apologize", "sorry,")

    for raw_line in text.strip().split("\n"):
        line = raw_line.strip()

        # Skip empty or very short lines
        if not line or len(line) < min_length:
            continue

        # Skip header-like lines (all caps and short)
        if line.isupper() and len(line) < 50:
            continue

        # Skip apologies if requested
        if skip_apologies and line.lower().startswith(apology_prefixes):
            continue

        insight: str | None = None

        # Check for bullet points
        if line.startswith(("-", "•", "*")):
            insight = line[1:].strip()
        # Check for numbered lists (1. or 1))
        elif (line[0].isdigit() and len(line) > 2 and line[1] in ".)") or (
            len(line) > 3 and line[:2].isdigit() and line[2] in ".)"
        ):
            # Find the separator position
            for i, char in enumerate(line):
                if char in ".)" and i < 4:
                    insight = line[i + 1 :].strip()
                    break
        # Accept plain text lines that look like complete statements
        elif len(line) > 30:
            insight = line

        if insight and len(insight) >= min_length:
            insights.append(insight)

    return insights
