from certamen.domain.interrogation.interrogator import (
    AdversarialInterrogator,
)


def test_parse_questions_numbered():
    text = "1. What is the evidence for your claim?\n2. Why did you ignore X?\n3. How does this work?"
    questions = AdversarialInterrogator._parse_questions(text)
    assert len(questions) == 3
    assert all("?" in q for q in questions)
    assert "What is the evidence for your claim?" in questions


def test_parse_questions_filters_non_questions():
    text = (
        "1. A statement without a question mark\n2. Is this a real question?"
    )
    questions = AdversarialInterrogator._parse_questions(text)
    assert len(questions) == 1
    assert "Is this a real question?" in questions


def test_parse_questions_empty():
    assert AdversarialInterrogator._parse_questions("") == []


def test_extract_answer_first_question():
    text = (
        "1. The answer to the first question is X.\n2. The second answer is Y."
    )
    answer = AdversarialInterrogator._extract_answer(1, text)
    assert "X" in answer
    assert "Y" not in answer


def test_extract_answer_last_question():
    text = "1. First answer.\n2. Second answer with detail."
    answer = AdversarialInterrogator._extract_answer(2, text)
    assert "Second answer" in answer


def test_extract_answer_fallback():
    text = "Just some text with no numbering."
    answer = AdversarialInterrogator._extract_answer(1, text)
    assert answer == text.strip()
