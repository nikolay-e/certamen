"""Utility functions for generating reports for benchmarks."""


def generate_manual_evaluation_template(model_names: list[str]) -> str:
    """Generates a markdown template for manual evaluation of benchmark results."""
    template = "## Manual Evaluation Guide\n\n"
    template += "Please evaluate all responses on these dimensions:\n\n"

    for i, model_name in enumerate(model_names):
        template += f"### {i + 1}. {model_name}\n"
        template += "- [ ] Technical Accuracy: Score 1-10\n"
        template += "- [ ] Completeness: Score 1-10\n"
        template += "- [ ] Nuance: Score 1-10\n"
        template += "- [ ] Actionability: Score 1-10\n"
        template += "- [ ] Overall Quality: Score 1-10\n\n"

    template += "### Which Would You Actually Use?\n\n"
    for model_name in model_names:
        template += f"- [ ] {model_name}\n"
    template += "- [ ] No significant difference\n\n"
    template += "### Why?\n"
    template += "(Write qualitative feedback here)\n\n"

    return template
