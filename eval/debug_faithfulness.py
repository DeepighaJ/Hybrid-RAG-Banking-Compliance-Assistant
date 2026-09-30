"""
Diagnostic: show exactly how the Ragas judge scores Faithfulness for ONE answer.

Re-scores a row from the most recent eval/results/eval_*.csv (no pipeline or
Azure calls, so it costs nothing beyond free-tier Groq judge calls) and prints
every raw judge output: first the list of claims extracted from the answer,
then the supported / not-supported verdict for each claim.

Usage:
    python eval/debug_faithfulness.py        # row 1 = second question
    python eval/debug_faithfulness.py 0      # row 0 = first question
"""

import ast
import glob
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

import pandas as pd
from langchain_core.callbacks import BaseCallbackHandler
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness
from ragas.run_config import RunConfig

from config import settings


class PrintJudgeOutputs(BaseCallbackHandler):
    """Print the raw text of every judge response."""

    def on_llm_end(self, response, **kwargs):
        for generations in response.generations:
            for gen in generations:
                print("\n===== JUDGE OUTPUT =====")
                print(gen.text)


def main() -> None:
    row_index = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    latest = sorted(glob.glob("eval/results/eval_*.csv"))[-1]
    row = pd.read_csv(latest).iloc[row_index]

    print(f"Results file: {latest}")
    print(f"Judge model:  {settings.GROQ_JUDGE_MODEL}")
    print(f"\nQUESTION:\n{row['user_input']}")
    print(f"\nANSWER BEING JUDGED:\n{row['response']}")

    sample = {
        "user_input": row["user_input"],
        "response": row["response"],
        "retrieved_contexts": ast.literal_eval(row["retrieved_contexts"]),
        "reference": row["reference"],
    }

    chat = ChatGroq(
        model=settings.GROQ_JUDGE_MODEL,
        api_key=settings.GROQ_API_KEY,
        callbacks=[PrintJudgeOutputs()],
    )

    result = evaluate(
        dataset=EvaluationDataset.from_list([sample]),
        metrics=[Faithfulness()],
        llm=LangchainLLMWrapper(chat),
        run_config=RunConfig(timeout=300, max_workers=1),
        show_progress=False,
    )
    print("\n===== FAITHFULNESS SCORE =====")
    print(result.to_pandas()["faithfulness"].iloc[0])


if __name__ == "__main__":
    main()