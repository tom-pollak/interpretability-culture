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
quizzes = dataset[[0]].clone()

# %%

def prep_quiz(quizzes, prefix_0=True, slice_at=304):
    if slice_at is not None:
        quizzes = quizzes[:, :slice_at]
    if prefix_0:
        quizzes = t.cat((
            t.zeros(quizzes.shape[0], 1, device=quizzes.device, dtype=quizzes.dtype),
            quizzes
        ), dim=1)
    return quizzes


# idxs = t.randint(dataset.size(0), (2,))

# %%

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


# import torch.nn.functional as F

# @t.inference_mode()
# def generate(
#     model,
#     input,
#     max_new_tokens,
#     temperature=1.0,
#     top_k=None,
#     deterministic=False,
# ):
#     model.eval()
#     for _ in range(max_new_tokens):
#         output, cache = model.run_with_cache(input)
#         logits = output[:, -1, :] / temperature

#         if top_k is not None:
#             v, _ = t.topk(logits, min(top_k, logits.size(-1)))
#             logits[logits < v[:, [-1]]] = -float('Inf')

#         probs = F.softmax(logits, dim=-1)

#         if deterministic:
#             next_token = t.argmax(probs, dim=-1)
#         else:
#             next_token = t.multinomial(probs, num_samples=1).squeeze(-1)

#         input = t.cat([input, next_token.unsqueeze(-1)], dim=-1)
#     return input


# gt_gen = generate(model, prep_quiz(quizzes[0][None]), 100, temperature=1)
# print(repr_grid(gt_gen[0, 1:]))

# # %%

# from transformer_lens import HookedTransformerKeyValueCache

# past_kv_cache = HookedTransformerKeyValueCache.init_cache(
#     model.cfg, model.cfg.device, 2
# )

# %%


from transformer_lens.utils import test_prompt

idx = 304 + 1
prompt = model.tokenizer.decode(prep_quiz(quizzes, slice_at=idx)[0])
answer = model.tokenizer.decode(quizzes[0][idx])

test_prompt(
    prompt,
    answer,
    model,
    prepend_space_to_answer=False,
    prepend_bos=False,
)


# %%


import circuitsvis as cv

logits, cache = model.run_with_cache(prompt)
cv.logits.token_log_probs(
    model.to_tokens(prompt),
    model(prompt)[0].log_softmax(dim=-1),
    model.to_string,
)

# %%

tokens = prep_quiz(quizzes, slice_at=403)
text = model.tokenizer.decode(tokens[0])
print(tokens.shape)

logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)

# %%
from IPython.display import display

attn_pattern = cache["pattern", 5]
attn_pattern = attn_pattern[:, -32:, -32:]
str_tokens = model.to_str_tokens(text)[-16:]

display(
    cv.attention.attention_patterns(
        tokens=str_tokens,
        attention=attn_pattern,
        # attention_head_names=[f"L0H{i}" for i in range(12)],
    )
)
# %%

print(cv.attention.attention_patterns(
    tokens=str_tokens,
    attention=attn_pattern,
    # attention_head_names=[f"L0H{i}" for i in range(12)],
))
