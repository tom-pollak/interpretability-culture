# %%

from interp import *
from interp.culture import *

import torch as t

from transformer_lens import HookedTransformer

device = get_device()

# %%
hts = load_hooked()
qm = load_quizzes()

model: HookedTransformer = hts[0].eval().to(device)  # type: ignore

# %%
gen_test_w_quiz_(model, qm, n=100)
dataset, from_w = qm.data_input(model, split="test")

# %%

out = model.generate(
    dataset[:1, :5],
    max_new_tokens=(100),
    stop_at_eos=False,
)
print(model.tokenizer.decode(out[0]))
# %%

from transformer_lens.utils import test_prompt

# Test the model with a prompt
test_prompt(
    "0 1 1 1 1",
    "1",
    model,
    prepend_space_to_answer=False,
)

# %%
