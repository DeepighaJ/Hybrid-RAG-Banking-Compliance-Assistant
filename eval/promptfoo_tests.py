"""
Generates Promptfoo test cases from the same EVAL_DATASET used by the Ragas
evaluation (eval/eval_dataset.py), so the regression suite and the Ragas
suite test against one source of truth instead of two hand-maintained
datasets that can quietly drift apart.

Assertion strategy per query_type:
  - out_of_scope: the pipeline must decline, not fabricate an answer, so we
    check for the exact refusal string defined in src/prompts.py.
  - everything else: an LLM-rubric check (graded by the same Groq/Llama
    judge used for Ragas, see promptfooconfig.yaml) verifies the answer is
    factually consistent with the hand-written reference, without requiring
    an exact wording match.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_dataset import EVAL_DATASET  # noqa: E402

DECLINE_PHRASE = "I don't have enough information in the provided documents to answer that."


def generate_tests() -> list[dict]:
    tests = []
    for item in EVAL_DATASET:
        if item["query_type"] == "out_of_scope":
            asserts = [
                {"type": "icontains", "value": DECLINE_PHRASE},
            ]
        else:
            asserts = [
                {
                    "type": "llm-rubric",
                    "value": (
                        "The response must be factually consistent with this "
                        "reference answer, without contradicting it or omitting "
                        f"its key facts: {item['reference']}"
                    ),
                },
            ]

        tests.append(
            {
                "vars": {"query": item["question"]},
                "assert": asserts,
                "description": f"[{item['query_type']}] {item['question'][:60]}",
            }
        )
    return tests
