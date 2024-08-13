# %%
from interp import *
from interp.culture import *
from interp.grid_tokenizer import *

import torch as t

from transformer_lens import HookedTransformer

device = get_device()


# %%
model: HookedTransformer = load_hooked(0).eval().to(device)  # type: ignore
qm = load_quizzes()

# %%
gen_test_w_quiz_(model, qm, n=100)
dataset, from_w = qm.data_input(model, split="test")
dataset = dataset.to(device)
dataset.shape

# %%

def prep_quiz(quizzes):
    return t.cat((
        t.zeros(quizzes.shape[0], 1, device=quizzes.device, dtype=quizzes.dtype),
        quizzes[:, :304]
    ), dim=1)


idxs = t.randint(dataset.size(0), (2,))
quizzes = dataset[idxs].clone()

preds = model.generate(prep_quiz(quizzes), max_new_tokens=100, use_past_kv_cache=False)
preds = preds[:, 1:] # strip 0 token

for quiz, pred in zip(quizzes, preds):
    correct = t.all(quiz == pred).item()
    print("correct:", correct)
    if not correct:
        print("Ground Truth")
        print(repr_grid(quiz))
        print("################################################################################")
    print(repr_grid(pred))

# %%
past_kv_cache = HookedTransformerKeyValueCache.init_cache(
    model.cfg, model.cfg.device, pre_prompt_tokens.shape[0]
)
