def detect_apology_or_refusal(response_text: str) -> bool:
    if not response_text:
        return False

    # Convert to lowercase for case-insensitive matching
    text_lower = response_text.lower().strip()

    # Common refusal/apology patterns
    refusal_patterns = [
        "i cannot",
        "i can't",
        "i'm sorry",
        "i am sorry",
        "i apologize",
        "sorry, i",
        "sorry but",
        "i'm unable",
        "i am unable",
        "i don't have",
        "i do not have",
        "as an ai",
        "i'm an ai",
        "i am an ai",
    ]

    # Check first 200 chars for refusal patterns
    # (refusals typically appear at the beginning)
    text_start = text_lower[:200]
    return any(pattern in text_start for pattern in refusal_patterns)
