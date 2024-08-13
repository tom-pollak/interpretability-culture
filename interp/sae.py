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

idx = t.randint(dataset.size(0), (1,)).item()
quizzes = dataset[idx][None].clone()

problem_struct = qm.problem.get_structure(quizzes)
for struct, mask, _ in qm.understood_structures:
    if struct == problem_struct:
        break

is_deterministic = sum(mask) == 1

print(f"world: {from_w[idx].item()} | struct: {struct} | mask: {mask} | deterministic: {is_deterministic}")

preds, correct = qm.predict(model, quizzes, struct, mask)
correct = correct.bool().item()
print("correct:", correct)

if not correct:
    print("Ground Truth")
    print(repr_grid(quizzes[0]))
    print("################################################################################")

print(repr_grid(preds[0]))


# %%

@t.inference_mode()
def one_batch_masked_inplace_autoregression(
    model,
    input,
    ar_mask,
    seq_logproba,
    deterministic_synthesis=False,
):
    to_generate = (ar_mask.sum(0) > 0).nonzero()
    for s in range(to_generate.min(), to_generate.max() + 1):
        output = model(TOK_PREPROCESS(input))

        logits = output[:, s]

        if deterministic_synthesis:
            t_next = logits.argmax(-1)
        else:
            dist = t.distributions.categorical.Categorical(logits=logits)
            t_next = dist.sample()

        all_n = t.arange(t_next.size(0))

        seq_logproba += logits[all_n, t_next]

        input[:, s] = ar_mask[:, s] * t_next + (1 - ar_mask[:, s]) * input[:, s]


ar_mask = qm.make_ar_mask(quizzes, struct, mask)
result = quizzes * (1 - ar_mask)
print(repr_grid(result[0]))


seq_logproba = t.empty(result.size(0), device=device)

one_batch_masked_inplace_autoregression(
    model,
    result,
    ar_mask,
    seq_logproba,
)

print(repr_grid(result[0]))

# %%
import torch.nn.functional as F

@t.inference_mode()
def generate(
    model,
    input,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    deterministic=False,
):
    model.eval()
    for _ in range(max_new_tokens):
        output = model(input)
        logits = output[:, -1, :] / temperature

        if top_k is not None:
            v, _ = t.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)

        if deterministic:
            next_token = t.argmax(probs, dim=-1)
        else:
            next_token = t.multinomial(probs, num_samples=1).squeeze(-1)

        input = t.cat([input, next_token.unsqueeze(-1)], dim=-1)
    return input


inp = t.cat((t.tensor([0], device=device), quizzes[~ar_mask.bool()])).unsqueeze(0)

gt_gen = generate(model, inp, 100, temperature=1)
hf_gen = model.generate(inp, temperature=1, max_new_tokens=99, use_past_kv_cache=False)

print(repr_grid(gt_gen[0, 1:]))
print("################################################################################")
print(repr_grid(hf_gen[0, 1:]))

# %%

print(generate(model, inp, 10, temperature=1)[0, -12:])
print(model.generate(inp, max_new_tokens=10, temperature=1, use_past_kv_cache=False)[0, -12:])

# %%


# print(struct, mask, problem_struct)
# struct, mask, _ = qm.understood_structures[0]
# reconfig_dset = qm.problem.reconfigure(dataset_subset, struct)
