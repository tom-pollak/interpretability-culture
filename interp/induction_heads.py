# %%
import torch as t
from transformer_lens import HookedTransformer

from interp.all import *

device = get_device()
model = load_hooked(0)
assert isinstance(model, HookedTransformer)
model.eval()
qm = load_quizzes()

tasks = qm.problem.all_tasks
task_names = [t.__name__ for t in tasks]
print("Available Tasks:\n- " + "\n- ".join(task_names) + "\n\n---------\n")

task_name = "task_frame"
num_tasks = 32
quizzes = qm.problem.generate_w_quizzes_(
    num_tasks, [qm.problem.all_tasks[task_names.index(task_name)]]
)
assert isinstance(quizzes, t.Tensor)
quizzes = quizzes.to(device)
print(repr_grid(quizzes[0]))


def zero_abl_hook(activation, hook):
    return t.zeros_like(activation)


final_grid_slice = slice(303, None)

# %%

model.add_hook("blocks.0.hook_attn_out", zero_abl_hook)  # type: ignore

preds = model.generate(
    prep_quiz(quizzes[:1]), max_new_tokens=100, use_past_kv_cache=False
)
assert isinstance(preds, t.Tensor)
preds = preds[:, 1:]  # strip 0 token

for quiz, pred in zip(quizzes, preds):
    correct = t.all(quiz == pred).item()
    print("correct:", correct)
    if not correct:
        print("Ground Truth")
        print(repr_grid(quiz))
        print(
            "################################################################################"
        )
    try:
        print(repr_grid(pred))
    except:
        print(preds[:, final_grid_slice])

model.reset_hooks()

# %%

"""

# Jumping to conclusions

So this was my firest indication that the model was incorporating an induction head in the first layer attention head

preds = [11, ... 12, ... 13, ... 14, 13, ... ]

Given 11, 12, 13 before, and 14 as the previous token, the model will consistently predict 13 as the next token (which is an invalid template)

This could indicate that the model uses the attention layer as a previous token circuit? And disabling the layer the model loses the ability to attend to the previous token?

The two ways (I know of) of creating an induction head are:

1. Using the previous token head through k-composition -- induction head: uses the current token as query, and attends highly to a matching previous token
2. Using the previous token head through q-composition -- induction head: the model uses the current token as the query, then the OV circuit rotates the positional embedding (in ROPE) to the next token

We're using sinusoidal positional encoding so might be harder (but not impossible) to rotate the positional embedding
"""

"""

# Multipying through the residual stream

Let's see which layers use the output of layer 0 the most. We can do this by multiplying through the residual stream

"""

# %%
