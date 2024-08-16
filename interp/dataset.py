from interp.culture import *

import torch as t
from datasets import Dataset


class DummyModel:
    pass


DATASET_NAME = "tommyp111/culture-puzzles-1M"
NELEMS = 1_000_000

if __name__ == "__main__":
    print(f"Creating {NELEMS} quizzes...")
    qm = load_quizzes()

    _dummy = DummyModel()
    gen_test_w_quiz_(_dummy, qm, n=NELEMS)
    tokens, from_w = qm.data_input(_dummy, split="test")

    tokens = t.cat(
        (t.zeros(tokens.shape[0], 1, device=tokens.device, dtype=tokens.dtype), tokens),
        dim=1,
    )

    print("tokens shape:", tokens.shape)
    print(f"% from culture: {(1 - from_w.float().mean().item())*100:.2f}")

    data_dict = {"input_ids": tokens, "from_w": from_w}

    dataset = Dataset.from_dict(data_dict)
    dataset.set_format(type="torch", columns=["input_ids", "from_w"])

    print("Pushing to hub...", end="")
    dataset.push_to_hub(DATASET_NAME)
    print("done.")
