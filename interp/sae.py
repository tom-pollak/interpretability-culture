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
dataset.shape

# %%
# out = model.generate(
#     dataset[:1, :5],
#     max_new_tokens=(100),
#     stop_at_eos=False,
# )
# print(model.tokenizer.decode(out[0]))

# %%
from transformer_lens.utils import test_prompt

# Test the model with a prompt
test_prompt(
    model.tokenizer.decode(dataset[0, :303], skip_special_tokens=False),
    "1",
    model,
    prepend_space_to_answer=False,
)

# %%
# qm = load_quizzes()
# qm.problem.save_some_examples(INTERP_RESULTS_DIR)

# # %%
# print(dataset[0][100:])

# # %%
# print(model.tokenizer.encode(model.tokenizer.decode(dataset[0, :303], skip_special_tokens=False), add_special_tokens=True))

# %%
# print(
#     model.generate(model.tokenizer.decode(dataset[0, :303]), max_new_tokens=101, stop_at_eos=False)
# )

# # %%
# for i in range(5):
#     print(
#         model.generate(
#             "0 D D D 0 1 1 E",
#             stop_at_eos=False,
#             temperature=1,
#             verbose=False,
#             max_new_tokens=(404-8),
#         )
#     )

# %%
