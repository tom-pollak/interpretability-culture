# %%
from interp.all import *
from datasets import load_dataset, Dataset, DatasetDict
from transformer_lens import HookedTransformer
from sae_lens import SAE
from pathlib import Path


device = get_device()
sae_pretrained_path = Path(__file__).parents[1] / "culture-gpt-0-sae"

model = load_hooked(0)
assert isinstance(model, HookedTransformer)
model = model.eval().to(device)


sae = SAE.load_from_pretrained(str(sae_pretrained_path), device=str(device))


dataset = load_dataset("tommyp111/culture-puzzles-1M", split="train")
assert isinstance(dataset, Dataset)
dataset.set_format("pt")


# %%
partitioned_datasets.push_to_hub("culture-puzzles-1M-partitioned")


# %%


@t.no_grad()
def quiz_diff_mask(quizzes):
    final_grid_slice = slice(
        -100, None
    )  # not including the special tokens (always different and easy)
    second_final_grid_slice = slice(-201, -101)
    return t.argwhere(
        quizzes[:, final_grid_slice] != quizzes[:, second_final_grid_slice]
    )


pd["frame"]
