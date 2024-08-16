# %%
import torch as t
from transformer_lens import HookedTransformer

from interp.culture import *
from interp import *
from interp.grid_tokenizer import *

device = get_device()
model = load_hooked(0)
assert isinstance(model, HookedTransformer)
qm = load_quizzes()

# %%

tasks = qm.problem.all_tasks
task_names = [t.__name__ for t in tasks]
print("Available Tasks:\n- " + "\n- ".join(task_names) + "\n\n---------\n")

task_name = "task_frame"
num_tasks = 32
quizzes = qm.problem.generate_w_quizzes_(
    num_tasks, [qm.problem.all_tasks[task_names.index(task_name)]]
).to(device)
assert isinstance(quizzes, t.Tensor)
print(repr_grid(quizzes[0]))


# %%
def zero_abl_hook(activation, hook):
    return t.zeros_like(activation)


final_grid_slice = slice(303, None)

model.eval()
task_losses = {}
for task_name, task in zip(task_names, tasks):
    print(f"MLP Zero Ablation ({task_name})")
    quizzes = qm.problem.generate_w_quizzes_(num_tasks, [task]).to(device)
    batch = TOK_PREPROCESS(quizzes)
    with t.inference_mode():
        orig_loss = (
            model(batch, return_type="loss", loss_per_token=True)[:, final_grid_slice]
            .mean()
            .item()
        )
        print(f"Orig: {orig_loss:.2e}")
        layer_diffs = []
        for i in range(model.cfg.n_layers):
            loss = (
                model.run_with_hooks(
                    batch,
                    return_type="loss",
                    fwd_hooks=[(f"blocks.{i}.hook_attn_out", zero_abl_hook)],
                    loss_per_token=True,
                )[:, final_grid_slice]
                .mean()
                .item()
            )
            print(f"Layer {i}: {loss:.2e}")
            layer_diffs.append((i, loss - orig_loss))
        task_losses[task_name] = layer_diffs
    print("\n------\n")


# %%

for task_name, layer_diffs in task_losses.items():
    layer_diffs.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop Layers ({task_name})")
    for i, diff in layer_diffs[:5]:
        print(f"\tLayer {i}: {diff:.2e}")

# %%

# ████████████████████████████████████████████████████████████████████████████████

allowed_layers = (0, 11)

fwd_hooks = [
    (f"blocks.{i}.hook_mlp_out", zero_abl_hook)
    for i in range(model.cfg.n_layers)
    if i not in allowed_layers
]

task_losses = {}
for task_name, task in zip(task_names, tasks):
    print(f"MLP Zero Ablation ({task_name})")
    quizzes = qm.problem.generate_w_quizzes_(num_tasks, [task]).to(device)
    batch = TOK_PREPROCESS(quizzes)
    with t.inference_mode():
        orig_loss = (
            model(batch, return_type="loss", loss_per_token=True)[:, final_grid_slice]
            .mean()
            .item()
        )
        layer_diffs = []
        loss = (
            model.run_with_hooks(
                batch,
                return_type="loss",
                fwd_hooks=fwd_hooks,
                loss_per_token=True,
            )[:, final_grid_slice]
            .mean()
            .item()
        )
        print(f"orig: {orig_loss:.2e} | ablated: {loss:.2e}")


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


# for hook in fwd_hooks:
#     model.add_perma_hook(*hook)
model.add_hook("blocks.0.hook_attn_out", zero_abl_hook)

preds = model.generate(prep_quiz(quizzes[:1]), max_new_tokens=100, use_past_kv_cache=False)
preds = preds[:, 1:] # strip 0 token

try:
    for quiz, pred in zip(quizzes, preds):
        correct = t.all(quiz == pred).item()
        print("correct:", correct)
        if not correct:
            print("Ground Truth")
            print(repr_grid(quiz))
            print("################################################################################")
        print(repr_grid(pred))
finally:
    model.reset_hooks()
    pass

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

Let's see which layers use the output of layer 0 the most

"""

# %%
