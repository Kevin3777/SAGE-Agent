"""Small, hand-authored multi-document QA tasks (not the HotpotQA benchmark)."""
from __future__ import annotations

import json
import re
import string
from typing import Any

from sage.core.types import Feedback, TaskCase


def _normalize(value: Any) -> str:
    text = str(value).casefold().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


class MultiHopQAEnvironment:
    def __init__(self) -> None:
        self.tasks = {
            "qa-ember-city": TaskCase(
                task_id="qa-ember-city", task_type="multi_hop_qa",
                user_task="What city is the studio behind Ember Calendar based in?",
                instructions=(
                    "Solve the two-hop chain: use p1 to identify the product's studio, then p2 "
                    "to find that studio's city. Return exactly one JSON object with string key "
                    "answer and list key evidence. The evidence list must contain BOTH passage "
                    "IDs p1 and p2, in that order; the Checker rejects an answer that cites only "
                    "the final location passage. Example shape: "
                    "{\"answer\":\"<city>\",\"evidence\":[\"p1\",\"p2\"]}. Do not add unsupported facts."
                ),
                payload={
                    "question": "What city is the studio behind Ember Calendar based in?",
                    "passages": {
                        "p1": "Ember Calendar was developed by Cedar Labs.",
                        "p2": "Cedar Labs has its headquarters in Wellington, New Zealand.",
                        "p3": "Harbor Notes was developed by Moss Studio.",
                    },
                    "answer": "Wellington",
                    "aliases": ["Wellington, New Zealand"],
                    "expected_evidence": ["p1", "p2"],
                },
                constraints=["Answer must be supported by both linking passages.", "Cite evidence passage IDs."],
            ),
            "qa-atlas-founder": TaskCase(
                task_id="qa-atlas-founder", task_type="multi_hop_qa",
                user_task="Who founded the company that created Atlas Notes?",
                instructions=(
                    "Follow both hops: use p1 to identify the company, then p2 to find its "
                    "founder. Return exactly one JSON object with string answer and list evidence. "
                    "The evidence list must contain BOTH p1 and p2, in that order. Do not cite "
                    "only the final passage."
                ),
                payload={
                    "question": "Who founded the company that created Atlas Notes?",
                    "passages": {
                        "p1": "Atlas Notes is a product of Northstar Labs.",
                        "p2": "Northstar Labs was founded by Imani Brooks in 2018.",
                        "p3": "Imani Brooks also advises the local library.",
                    },
                    "answer": "Imani Brooks",
                    "aliases": [],
                    "expected_evidence": ["p1", "p2"],
                },
                constraints=["Answer must follow the product-to-company-to-founder chain.", "Cite evidence passage IDs."],
            ),
        }

    def get_task(self, task_id: str = "qa-ember-city") -> TaskCase:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise ValueError(f"unknown QA task {task_id!r}; available: {', '.join(self.tasks)}") from exc

    @staticmethod
    def format_prompt(task: TaskCase) -> str:
        passages = "\n".join(f"[{key}] {value}" for key, value in task.payload["passages"].items())
        return f"Question: {task.user_task}\nPassages:\n{passages}"

    @staticmethod
    def parse_output(raw: str) -> dict[str, Any]:
        raw = raw.strip()
        if not raw:
            raise ValueError("empty assistant output")
        candidates = [raw]
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            candidates.insert(0, match.group(0))
        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "answer" in value:
                evidence = value.get("evidence", value.get("citations", []))
                if isinstance(evidence, (str, int)):
                    evidence = [evidence]
                if not isinstance(evidence, list):
                    evidence = []
                evidence_ids = []
                for item in evidence:
                    evidence_ids.extend(re.findall(r"\bp\d+\b", str(item), flags=re.IGNORECASE))
                return {
                    "answer": str(value["answer"]).strip(),
                    "evidence": list(dict.fromkeys(item.casefold() for item in evidence_ids)),
                }
        answer = re.search(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", raw)
        evidence = re.findall(r"\bp\d+\b", raw, flags=re.IGNORECASE)
        if answer:
            return {"answer": answer.group(1).strip(), "evidence": evidence}
        return {"answer": raw, "evidence": evidence}

    def check(self, task: TaskCase, candidate: Any) -> Feedback:
        if not isinstance(candidate, dict) or "answer" not in candidate:
            error = "output could not be parsed into an answer object"
            return Feedback.create(
                False, 0.0, f"Malformed answer: {error}. Return JSON with answer and evidence.",
                [error], [f"checker:{task.task_id}:format"],
            )
        expected = [task.payload["answer"], *task.payload.get("aliases", [])]
        answer_ok = any(_normalize(candidate.get("answer", "")) == _normalize(item) for item in expected)
        cited = {str(value).casefold() for value in candidate.get("evidence", [])}
        required = set(task.payload["expected_evidence"])
        evidence_ok = required.issubset(cited)
        errors: list[str] = []
        if not answer_ok:
            errors.append(
                f"answer mismatch: the linking passages support {task.payload['answer']!r}, "
                f"but the output gave {candidate.get('answer')!r}"
            )
        if not evidence_ok:
            missing = sorted(required - cited)
            errors.append(
                f"missing required supporting passages: {', '.join(missing)}; "
                f"include both passage IDs {', '.join(sorted(required))} in the evidence list"
            )
        if errors:
            message = "Checker rejected the answer. " + "; ".join(errors) + "."
            score = (0.65 if answer_ok else 0.0) + (0.35 if evidence_ok else 0.0)
            return Feedback.create(False, score, message, errors, list(task.payload["expected_evidence"]))
        return Feedback.create(
            True, 1.0, "Correct answer and both passage links were verified.",
            [], list(task.payload["expected_evidence"]),
        )
