_APOLOGY_PREFIXES = ("i cannot", "i'm sorry", "i apologize", "sorry,")


def _should_skip_line(
    line: str, min_length: int, skip_apologies: bool
) -> bool:
    if not line or len(line) < min_length:
        return True
    if line.isupper() and len(line) < 50:
        return True
    if skip_apologies and line.lower().startswith(_APOLOGY_PREFIXES):
        return True
    return False


def _extract_bullet_insight(line: str) -> str | None:
    if line.startswith(("-", "•", "*")):
        return line[1:].strip()
    return None


def _extract_numbered_insight(line: str) -> str | None:
    is_single_digit = line[0].isdigit() and len(line) > 2 and line[1] in ".)"
    is_double_digit = len(line) > 3 and line[:2].isdigit() and line[2] in ".)"
    if not (is_single_digit or is_double_digit):
        return None
    for i, char in enumerate(line):
        if char in ".)" and i < 4:
            return line[i + 1 :].strip()
    return None


def _extract_insight(line: str) -> str | None:
    bullet = _extract_bullet_insight(line)
    if bullet is not None:
        return bullet
    numbered = _extract_numbered_insight(line)
    if numbered is not None:
        return numbered
    if len(line) > 30:
        return line
    return None


def parse_insight_lines(
    text: str,
    min_length: int = 10,
    skip_apologies: bool = True,
) -> list[str]:
    insights = []

    for raw_line in text.strip().split("\n"):
        line = raw_line.strip()

        if _should_skip_line(line, min_length, skip_apologies):
            continue

        insight = _extract_insight(line)

        if insight and len(insight) >= min_length:
            insights.append(insight)

    return insights
