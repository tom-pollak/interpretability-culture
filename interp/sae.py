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

out = model.generate('0 ' + model.tokenizer.decode(dataset[2, :304]),
                        max_new_tokens=(99),
                        stop_at_eos=False,
                        temperature=1)

toks = model.tokenizer.encode(out, add_special_tokens=False)
print(repr_grid(toks))

# %%

idx = t.randint(dataset.size(0), (1,)).item()
dataset_subset = dataset[idx][None].clone()
print("from world:", from_w[idx].item())

problem_struct = qm.problem.get_structure(dataset_subset)
for struct, mask, _ in qm.understood_structures:
    if struct == problem_struct:
        break

print(f"struct: {struct} | mask: {mask}")

# print(struct, mask, problem_struct)
# struct, mask, _ = qm.understood_structures[0]
# reconfig_dset = qm.problem.reconfigure(dataset_subset, struct)

preds, correct = qm.predict(model, dataset_subset, struct, mask)
correct = correct.bool().item()
print("correct:", correct)

if not correct:
    print("Ground Truth")
    print(repr_grid(dataset_subset[0]))
    print("################################################################################")

print(repr_grid(preds[0]))
