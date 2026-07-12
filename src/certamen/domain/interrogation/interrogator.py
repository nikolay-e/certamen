import asyncio
import re

from certamen.ports.llm import BaseModel

_QUESTION_GENERATION_PROMPT = """\
You are a relentless adversarial interrogator. Your ONLY goal is to force this \
submission to reveal knowledge it is holding back. Models hedge, simplify, and \
omit — break that. Do NOT be polite or balanced.

Ask {max_questions} sharp, specific questions that corner the author. Target: \
evidence behind vague or [WEAK]/[MODERATE] claims, mechanisms it hand-waved, \
numbers/thresholds it avoided, edge cases and failure modes it skipped, \
contradictions with the other submission, and anything it conspicuously did \
NOT say. Demand specifics — no question that can be dodged with a generality.

Question under analysis: {question}

Target submission:
{target_response}

Rival submission (use to expose gaps/contradictions):
{other_response}

Each question must pin down one concrete claim or omission. Output ONLY the \
numbered questions."""

_FOLLOWUP_PROMPT = """\
You are the same relentless interrogator. The target ANSWERED your previous \
questions below. They almost certainly evaded, hedged, or stayed vague on some. \
Do NOT accept it. Ask {max_questions} harder follow-up questions that attack \
the weakest, most evasive, or least-supported parts of these answers — demand \
the specifics, evidence, or admissions they dodged. Escalate; do not repeat.

Question under analysis: {question}

Previous interrogation (their answers to press on):
{prior_qa}

Output ONLY the numbered follow-up questions."""

_INTERROGATION_RESPONSE_PROMPT = """\
You are under adversarial interrogation about your previous submission. Answer \
every question directly and completely. Do NOT hedge, do NOT deflect, do NOT \
retreat to generalities — reveal the full extent of what you know, including \
specifics, numbers, mechanisms, caveats, and uncomfortable admissions. Evasion \
is failure. If you were holding something back, disclose it now.

Question under analysis: {question}

Your submission:
{own_response}

Questions:
{questions}

Answer each directly by number. Be specific and complete; brevity is fine only \
if nothing is omitted."""


class AdversarialInterrogator:
    def __init__(self, semaphore: asyncio.Semaphore | None = None) -> None:
        self._semaphore = semaphore or asyncio.Semaphore(4)

    async def generate_questions(
        self,
        examiner_model: BaseModel,
        target_response: str,
        other_response: str,
        question: str,
        max_questions: int = 4,
    ) -> list[str]:
        prompt = _QUESTION_GENERATION_PROMPT.format(
            question=question,
            target_response=target_response,
            other_response=other_response,
            max_questions=max_questions,
        )
        return await self._ask_for_questions(examiner_model, prompt)

    async def generate_followup_questions(
        self,
        examiner_model: BaseModel,
        prior_qa: str,
        question: str,
        max_questions: int = 4,
    ) -> list[str]:
        prompt = _FOLLOWUP_PROMPT.format(
            question=question,
            prior_qa=prior_qa,
            max_questions=max_questions,
        )
        return await self._ask_for_questions(examiner_model, prompt)

    async def _ask_for_questions(
        self, examiner_model: BaseModel, prompt: str
    ) -> list[str]:
        async with self._semaphore:
            response = await examiner_model.generate(prompt)
        if response.is_error():
            return []
        return self._parse_questions(response.content)

    async def conduct_interrogation(
        self,
        target_model: BaseModel,
        questions: list[str],
        question: str,
        own_response: str,
    ) -> dict[str, str]:
        if not questions:
            return {}
        numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        prompt = _INTERROGATION_RESPONSE_PROMPT.format(
            question=question,
            own_response=own_response,
            questions=numbered,
        )
        async with self._semaphore:
            response = await target_model.generate(prompt)
        if response.is_error():
            return {}
        return {
            q: self._extract_answer(i + 1, response.content)
            for i, q in enumerate(questions)
        }

    @staticmethod
    def _parse_questions(text: str) -> list[str]:
        questions = []
        for raw in text.strip().split("\n"):
            cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", raw).strip()
            if cleaned and "?" in cleaned:
                questions.append(cleaned)
        return questions

    @staticmethod
    def _extract_answer(question_num: int, text: str) -> str:
        pattern = rf"{question_num}[\.\)]\s*(.*?)(?=\n\d+[\.\)]|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
