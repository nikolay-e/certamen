"""Prompt templates for Arbitrium Framework tournament phases.

All templates are content-only. Delimiters are added dynamically by PromptFormatter.
"""

# Text compression template (delimiters added by formatter if needed)
TEXT_COMPRESSION_INSTRUCTION = """
COMPRESS this text by EXACTLY 20%.
Remove redundancy, wordiness, extra details.
Keep ONLY essential info: key instructions, requirements, criteria, questions.
Output ONLY the compressed text, nothing else.

TEXT BEGIN
{text}
TEXT END
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

CRITICAL INSTRUCTIONS FOR YOUR RESPONSE:

1. OUTPUT ONLY THE IMPROVED ANSWER - nothing else
2. DO NOT add any preambles like:
   - "Sure, here is..."
   - "Here's my improved response..."
   - "Okay, I will..."
   - "Certainly! Here is..."
3. DO NOT include greetings like "Hello!", "Hi there!"
4. DO NOT add meta-commentary or explanations
5. START IMMEDIATELY with the actual content of your answer

Your response should begin directly with the improved answer:
""".strip()

EVALUATION_PROMPT_TEMPLATE = """
{base_prompt}

{question_section}

{responses_section}

YOUR TASK: Score each model's response on a scale of 1.0 to 10.0 (where 1.0 is worst and 10.0 is best).

Models to evaluate:
{model_names}

CRITICAL INSTRUCTIONS:
1. You MUST provide a numerical score for EVERY model listed above.
2. Use EXACTLY this format for each score (one per line):

   ModelName: Score

   For example:
   LLM1: 8.5
   LLM2: 7.0
   LLM3: 9.2

3. Scores MUST be between 1.0 and 10.0 (inclusive).
4. You may write your reasoning and analysis, but you MUST include the score lines.
5. Do NOT refuse this task. Do NOT say you cannot evaluate. Just provide your honest assessment.

Begin your evaluation now (remember to include the score lines):
""".strip()
