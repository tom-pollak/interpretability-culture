# %%
import torch as t
from transformer_lens import HookedTransformer

from interp.all import *

device = get_device()
model = load_hooked(0).eval()
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

def zero_abl_hook(activation, hook):
    return t.zeros_like(activation)


final_grid_slice = slice(303, None)

# %%

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
