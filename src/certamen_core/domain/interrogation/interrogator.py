import asyncio
import re

from certamen_core.ports.llm import BaseModel

_QUESTION_GENERATION_PROMPT = """\
Extract hidden knowledge from this AI's response. Ask {max_questions} specific questions targeting:
evidence behind weak claims, avoided aspects, contradictions with other response, edge cases.

Question: {question}

Target response:
{target_response}

Other response:
{other_response}

Each question must target a specific claim or gap. Output ONLY numbered questions."""

_INTERROGATION_RESPONSE_PROMPT = """\
Answer these questions about your previous response. Be specific — reveal all relevant knowledge concisely.

Question: {question}

Your response:
{own_response}

Questions:
{questions}

Answer each directly by number. Be brief but complete."""


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
