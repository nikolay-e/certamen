# Text compression template (delimiters added by formatter if needed)
TEXT_COMPRESSION_INSTRUCTION = """
COMPRESS by 20%. Remove redundancy and wordiness. Keep essential info only. Output ONLY compressed text.

{text}
""".strip()

# Evaluator response template (simple format, no delimiters needed)
LOG_EVALUATOR_RESPONSE = """
Evaluator {evaluator} response:
{content}
""".strip()

# Full prompt templates for tournament phases
# Note: Section wrapping (BEGIN/END delimiters) is done by PromptFormatter
INITIAL_PROMPT_TEMPLATE = """
{base_prompt}

{question_section}
""".strip()

FEEDBACK_PROMPT_TEMPLATE = """
{feedback_instruction}

{base_prompt}

{question_section}

{answer_section}
""".strip()

IMPROVEMENT_PROMPT_TEMPLATE = """
{improvement_instruction}

{base_prompt}

{question_section}

{answer_section}
{context_section}{knowledge_section}

OUTPUT ONLY the improved answer. No preambles, greetings, or meta-commentary. Start directly with content.
""".strip()

SYNTHESIS_PROMPT_TEMPLATE = """
{synthesis_instruction}

{question_section}

{all_responses_section}
{knowledge_section}

OUTPUT ONLY the synthesized answer. No preambles, greetings, or meta-commentary. Start directly with content.
""".strip()

EVALUATION_PROMPT_TEMPLATE = """
{base_prompt}

{question_section}

{responses_section}

Score each model 1.0-10.0. Models: {model_names}

Format EXACTLY as: ModelName: Score (e.g., LLM1: 8.5)
You MUST score EVERY model. Brief reasoning, then scores.
""".strip()
