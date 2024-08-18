# %%
# fmt: off
import torch as t
from transformer_lens import HookedTransformer, ActivationCache
import circuitsvis as cv
from IPython.display import display

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

print("Running zero ablation on attention...")

verbose = False
k = 5
task_losses = {}
orig_losses = {}
for task_name, task in zip(task_names, tasks):
    if verbose:
        print(f"# {task_name}")
    quizzes = qm.problem.generate_w_quizzes_(num_tasks, [task]).to(device)
    batch = TOK_PREPROCESS(quizzes)
    with t.inference_mode():
        orig_loss = (
            model(batch, return_type="loss", loss_per_token=True)[:, final_grid_slice]
            .mean()
            .item()
        )
        if verbose:
            print(f"Orig: {orig_loss:.2e}")
        orig_losses[task_name] = orig_loss
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
            if verbose:
                print(f"Layer {i}: {loss:.2e}")
            layer_diffs.append((i, loss - orig_loss))
        task_losses[task_name] = layer_diffs
    if verbose:
        print("\n------\n")

for task_name, layer_diffs in task_losses.items():
    layer_diffs.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTopk layers with highest loss: ({task_name})")
    print(f"\tOrig: {orig_losses[task_name]:.2e}")
    for i, diff in layer_diffs[:k]:
        print(f"\tdiff {i}: {diff:.2e}")

"""
Wow! we can see that the first (0th) layer has an outsized impact on the loss of *all* tasks. All other layers have a several order of magnitude smaller impact.
"""

# %%

model.add_hook("blocks.0.hook_attn_out", zero_abl_hook)  # type: ignore
generate_and_print(model, quizzes[0])
model.reset_hooks()

# %%

"""

# Jumping to conclusions

This was my first indication that the model was incorporating an induction head in the first layer attention head

preds = [11, ... 12, ... 13, ... 14, 13, ... ]

Given 11, 12, 13 before, and 14 as the previous token, the model will consistently predict 13 as the next token (which is an invalid template)

This could indicate that the model uses the attention layer as a previous token circuit? And disabling the layer the model loses the ability to attend to the previous token?

Let's see if we can first find the previous token head
"""

quiz = quizzes[None, 0]
str_tokens = model.to_str_tokens(quiz)

logits, cache = model.run_with_cache(quiz, remove_batch_dim=True)

def prev_attn_detector(cache, thres, n_lookback):
    out = []
    for layer in range(GPT_SMALL.n_layers):
        attn_pattern = cache["pattern", layer].cpu()
        diag = attn_pattern[
            :,
            t.arange(GPT_SMALL.n_ctx),
            t.clamp(t.arange(GPT_SMALL.n_ctx) - n_lookback, 0)
        ]
        mean_diag = diag.mean(dim=1)
        mask = mean_diag > thres
        heads, values = t.nonzero(mask, as_tuple=True)[0], mean_diag[mask]
        out.extend([f"{layer}.{head.item()}: {value.item():.4f}"
                    for head, value in zip(heads, values)])
    return out

print("Heads attending to previous token =", ", ".join(prev_attn_detector(cache, thres=0.2, n_lookback=1)))

# %%

"""

Hm! there's only one head that attends to the previous token in layer 0, and the value is quite low.

In fact there's a much stronger head in layer 7

"""


attn_pattern = cache["pattern", 0][7]

display(
    cv.attention.attention_pattern(
        tokens=str_tokens,
        attention=attn_pattern,
    )
)

# %%
"""
Let's look back a whole grid then:
"""

print("Heads attending to previous grid =", ", ".join(prev_attn_detector(cache, thres=0.1, n_lookback=100)))

# %%

"""
There's an even stronger head, 0!

And we can see that it strongly relates the to the current token from the previous grid, and if in the first grid it attends to the previous token.

(It also slightly attends to the token 20 tokens either side of the previous grid)

This could be a fixed distance induction head???? Of course the model will know *exactly where the previous token will be! Always 100 tokens back.
"""

attn_pattern = cache["pattern", 0][0]

display(
    cv.attention.attention_pattern(
        tokens=str_tokens,
        attention=attn_pattern,
    )
)

# %%

"""
Ok let's try to test this, if the attention pattern writes directly into W_U subspace, we can ablate ALL other layers & heads, and see if the output repeats the previous grid.

This isn't foolproof, I'm sure there must be a lot more in the network, but it's a nice test to see if we're on track.

So we're essentially doing: W_E (layer 0 head 0) W_U
"""


ablate_mlp = [
    (f"blocks.{i}.hook_mlp_out", zero_abl_hook)
    for i in range(model.cfg.n_layers)
]


def zero_abl_except_head(activation, hook):
    B, H, T, _ = activation.shape
    act = t.zeros(B, H, T, T)
    act[:, :, t.arange(T), t.clamp(t.arange(T)-100, 0)] = 1
    return act

ablate_attn = [
    (f"blocks.{i}.attn.hook_pattern",
    zero_abl_except_head if i == 0 else zero_abl_hook)
    for i in range(model.cfg.n_layers)
]


for hook in ablate_mlp + ablate_attn:
    model.add_hook(*hook)  # type: ignore

generate_and_print(model, quiz)
model.reset_hooks()

# %%

"""
Ok so it's not perfect, but it's pretty close!

We could go further down this rabbit hole, and try to extract a "clean" representation of the previous grid, but this seems fairly trivial and got me thinking -- for 90% of even the new grid, the model can just copy the previous grid. What seems more interesting is the 10% of the time when the model does something different.

It makes sense that even though it has a much smaller impact on the overall loss, the calculations done in later layers are much more interesting.
"""



for layer in range(GPT_SMALL.n_layers):
    print(f"ablating attn layer {layer}: ", end="")
    model.add_hook(f"blocks.{layer}.hook_attn_out", zero_abl_hook)  # type: ignore
    correct, pred = generate(model, quizzes[:1])
    print("correct:", correct.item())
    if not correct:
        print("Ground Truth")
        print(repr_grid(quizzes[0, 303:]))
        print("Predicted")
        print(repr_grid(pred[0, 303:]))
    model.reset_hooks()

model.reset_hooks()

# %%

"""
Instead of investigating the whole predicted grid, let's instead investigate the difference between the predicted grid and the previous grid. Maybe there's more interesting features!
"""

print("Running zero ablation on attention...")

@t.no_grad()
def quiz_diff_mask(quizzes):
    final_grid_slice = slice(-100, None) # not including the special tokens (always different and easy)
    second_final_grid_slice = slice(-201, -101)
    return t.argwhere(quizzes[:, final_grid_slice] != quizzes[:, second_final_grid_slice])


task_name = "task_frame"
num_tasks = 32
quizzes = qm.problem.generate_w_quizzes_(
    num_tasks, [qm.problem.all_tasks[task_names.index(task_name)]]
)
assert isinstance(quizzes, t.Tensor)
quizzes = quizzes.to(device)
batch = TOK_PREPROCESS(quizzes)
mask = quiz_diff_mask(quizzes)
with t.inference_mode():
    orig_logits, orig_loss = model(batch, return_type="both", loss_per_token=True)
    orig_tfm_loss = orig_loss[mask].mean().item()
    orig_tfm_logits = orig_loss[mask].mean().item()
    print(f"orig loss: {orig_tfm_loss:.2e}")
    for i in range(model.cfg.n_layers):
        logits, loss = model.run_with_hooks(
            batch,
            return_type="both",
            fwd_hooks=[(f"blocks.{i}.hook_attn_out", zero_abl_hook)],
            loss_per_token=True,
        )
        tfm_loss = loss[mask].mean().item()
        tfm_logits = logits[mask]
        print(f"Layer {i} diff: {tfm_loss - orig_tfm_loss:.2e}")

# %%




"""

The two ways (I know of) of creating an induction head are:

1. Using the previous token head through k-composition -- induction head: uses the current token as query, and attends highly to a matching previous token
2. Using the previous token head through q-composition -- induction head: the model uses the current token as the query, then the OV circuit rotates the positional embedding (in ROPE) to the next token

We're using sinusoidal positional encoding so might be harder (but not impossible) to rotate the positional embedding

"""


# %%

"""

# Multipying through the residual stream

Let's see which layers use the output of layer 0 the most. We can do this by multiplying through the residual stream

"""

# %%
