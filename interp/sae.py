# %%
from interp.all import *
from datasets import load_dataset, DatasetDict
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer
from sae_lens import SAE
import json
from pathlib import Path
from functools import partial
import torch as t


device = get_device()
sae_pretrained_path = Path(__file__).parents[1] / "culture-gpt-0-sae"

model: HookedTransformer
model = load_hooked(0).eval().to(device)  # type: ignore


def load_sae(path: str) -> SAE:
    from sae_lens import SAEConfig
    from sae_lens.toolkit.pretrained_sae_loaders import read_sae_from_disk
    from sae_lens.config import DTYPE_MAP

    weight_path = hf_hub_download(SAE_REPO_ID, path + "/sae_weights.safetensors")
    cfg_path = hf_hub_download(SAE_REPO_ID, path + "/cfg.json")
    with open(cfg_path, "r") as f:
        cfg_dict = json.load(f)
    cfg_dict, state_dict = read_sae_from_disk(
        cfg_dict=cfg_dict,
        weight_path=weight_path,
        device="cpu",
        dtype=DTYPE_MAP[cfg_dict["dtype"]],
    )
    print(cfg_dict)
    sae_cfg = SAEConfig.from_dict(cfg_dict)
    sae_cfg.device = str(device)
    sae = SAE(sae_cfg)
    sae.load_state_dict(state_dict)
    sae.eval()
    return sae


sae = load_sae("gpt-0/blocks_8_mlp_out")

dataset = load_dataset("tommyp111/culture-puzzles-1M-partitioned")
assert isinstance(dataset, DatasetDict)
dataset.set_format("pt")

# %%

"""

Bit of a util for later, The first and second grid are a one-shot example, so we should not include these in our eval / analysis. Let's create slices that extract the final and second final grid.

(not including the special tokens A f(A) etc. These are easy).

Another gotcha: these should be from a "stripped" batch, i.e. a batch without it's final token. I've found this edge case to be rather strange, as


I've found this edge case to be rather strange, and HookedTransformer complains about


"""


final_grid_slice = slice(-99, None)
second_final_grid_slice = slice(-200, -101)

batch = dataset["contact"]["input_ids"][:5].to(device)
batch_stripped = batch[:, :-1]

a = batch_stripped[:1].clone()
print(repr_grid(a[0]))
print("\n████████████████████████████████████████████████████████████████████████████████\n")
a[:, second_final_grid_slice] = 4
a[:, final_grid_slice] = 5
print(repr_grid(a[0]))

# %%

"Now let's "


def ablate_mlp_single_task(batch, layer):
    batch_stripped = batch[:, :-1]
    with t.inference_mode():
        logits_orig, loss_orig = model(batch_stripped, return_type="both")
        logits_orig = logits_orig[:, final_grid_slice]

        logits_abl, loss_abl = model.run_with_hooks(
            batch_stripped,
            return_type="both",
            fwd_hooks=[(f"blocks.{layer}.hook_mlp_out", zero_abl_hook)],
        )
        logits_abl = logits_abl[:, final_grid_slice]

    # argmax => temp 0
    orig_correct = t.all(batch[:, final_grid_slice] == logits_orig.argmax(-1), dim=1)
    abl_same = t.all(logits_orig.argmax(-1) == logits_abl.argmax(-1), dim=1)
    wrong_puzzles = (~orig_correct).argwhere()[:, 0].tolist()
    abl_diff_puzzles = (~abl_same).argwhere()[:, 0].tolist()

    wrong_and_different_ablate = list(
        set(wrong_puzzles).intersection(set(abl_diff_puzzles))
    )
    wrong_because_ablate = list(set(abl_diff_puzzles).difference(set(wrong_puzzles)))

    print("wrong puzzles:", wrong_puzzles)
    print("different w/ ablation:", abl_diff_puzzles)
    print("different abaltation & wrong:", wrong_and_different_ablate)
    print("wrong because of ablation:", wrong_because_ablate)
    print()
    return wrong_because_ablate


# %%

n = 100
layer = 8
for task, data in dataset.items():
    print("task:", task)
    batch = data[:n]["input_ids"].to(device)
    ablate_mlp_single_task(batch, layer)
# %%
"""
Wow! Ablating layer 8 seems to primarily only impact only the contact task, let's have a look at the quizzes the model got wrong with ablation

Culture is a mix of all tasks, so
"""
n = 100
layer = 8
batch = dataset["contact"][:n]["input_ids"].to(device)
wrong_because_ablate = ablate_mlp_single_task(batch, layer)

for i in range(min(5, len(wrong_because_ablate))):
    model.add_hook("blocks.8.hook_mlp_out", zero_abl_hook)  # type: ignore
    generate_and_print(model, batch[wrong_because_ablate[i]], temperature=0.0)
    model.reset_hooks()

# %%
"""
All the examples seem to fall victim to the same problem!

# Problem

Given the one-shot example, one of the squares are contacted, and encircles the other square with their color.

When this layer is ablated, it seems to lose the ability to lookback and see which square should be encircled, and instead incircles both!


Using train_sae.py, I trained a model on layer 8. This uses all data in tommyp111/culture-puzzles-1M, not just this particular task. Let's see if applying the SAE weights can resolve this problem for these tasks
"""

model.reset_hooks()
with t.inference_mode():
    _, cache = model.run_with_cache(batch[wrong_because_ablate, :-1])
    h = cache[sae.cfg.hook_name].clone()
    del cache
    feature_acts = sae.encode(h)
    del h
    sae_out = sae.decode(feature_acts)

# %%


def reconstr_hook(activation, hook, sae_out):
    _, T, _ = activation.shape
    return sae_out[None, :T, :]


model.reset_hooks()
tasks_correct = []
for i in range(min(5, len(wrong_because_ablate))):
    model.add_hook(sae.cfg.hook_name, partial(reconstr_hook, sae_out=sae_out[i]))  # type: ignore
    correct, _ = generate_and_print(
        model, batch[wrong_because_ablate[i]], temperature=0.0
    )
    tasks_correct.append(correct)
    model.reset_hooks()

print("All tasks correct:", all(tasks_correct))
"""

Yay it does! The SAE fixes the behaviour on all quizzes that the model failed on with zero ablation, this could be a good indicator that my SAE is working correctly :)

"""

# %%

"""
Let's take a look at the loss w.r.t normal, zero and reconstruction for the all tasks and the contact task specifically.

Like before, I'm going to look at the loss at only the final grid where the square is different to the square from the previous grid -- the "transform" squares
"""


# %%

"""
Investiage which W_dec neurons fire
"""

# %%

"""
Find the mlp in the other models that cause this same behaviour -- is this univeral neurons?
"""

# %%

# %%
# ████████████████████████████████████  old  █████████████████████████████████████

"Instead of investigating the whole predicted grid, let's instead investigate the difference between the predicted grid and the previous grid. Maybe there's more interesting features!"
"Nifty, now let's create a mask that selects the squares that are different between the final two grids (colored yellow)"

@t.no_grad()
def quiz_diff_mask(quizzes):
    final_grid_slice = slice(-99, None)
    second_final_grid_slice = slice(-200, -101)

    idxs = t.argwhere(
        quizzes[:, final_grid_slice] != quizzes[:, second_final_grid_slice]
    )
    B, T = quizzes.shape
    start_idx = T + final_grid_slice.start
    idxs[:, 1] += start_idx
    return idxs

def get_mask(batch, mask):
    return batch[mask[:, 0], mask[:, 1]]

idxs = quiz_diff_mask(batch_stripped)
a = batch_stripped.clone()
print(repr_grid(a[1]))
print("\n████████████████████████████████████████████████████████████████████████████████\n")
# get_mask(a, idxs).fill_(4)
a[idxs[:,0], idxs[:,1]] = 4
print(repr_grid(a[1]))


def zero_abl_hook(activation, hook):
    return t.zeros_like(activation)

for task, data in dataset.items():
    print("\ntask:", task)
    batch = data["input_ids"][:5].to(device)
    batch_stripped = batch[:, :-1]
    mask = quiz_diff_mask(batch_stripped)
    with t.inference_mode():
        model.reset_hooks()
        orig_logits, orig_loss = model(
            batch_stripped, return_type="both", loss_per_token=True
        )

        # orig_tfm_loss = orig_loss[mask[:,0], mask[:,1]].mean().item()
        # orig_tfm_logits = orig_logits[mask[:,0], mask[:,1]].mean().item()

        orig_tfm_loss = orig_loss[:, final_grid_slice].mean().item()
        orig_tfm_logits = orig_logits[:, final_grid_slice].mean().item()

        # orig_tfm_loss = orig_loss.mean().item()
        # orig_tfm_logits = orig_loss.mean().item()

        print(f"orig loss: {orig_tfm_loss:.2e}")
        for i in range(model.cfg.n_layers):
            logits, loss = model.run_with_hooks(
                batch_stripped,
                return_type="both",
                fwd_hooks=[(f"blocks.{i}.hook_mlp_out", zero_abl_hook)],
                loss_per_token=True,
            )
            # tfm_loss = loss[mask[:,0], mask[:,1]].mean().item()
            # tfm_logits = logits[mask[:,0], mask[:,1]]

            tfm_loss = loss[:, final_grid_slice].mean().item()
            tfm_logits = logits[:, final_grid_slice].mean().item()

            # tfm_loss = loss.mean().item()
            # tfm_logits = logits
            print(f"Layer {i} diff: {tfm_loss - orig_tfm_loss:.2e}")



# ███████████████████████████████████  random  ███████████████████████████████████


# from sae_lens import SAE

# sae = SAE.load_from_pretrained(str(pretrained_path), device=device)

# # %%

# import torch

# import pandas as pd

# # Let's start by getting the top 10 logits for each feature
# projection_onto_unembed = sae.W_dec @ MODEL.W_U


# # get the top 10 logits.
# vals, inds = torch.topk(projection_onto_unembed, 10, dim=1)

# # get 10 random features
# random_indices = torch.randint(0, projection_onto_unembed.shape[0], (10,))

# # Show the top 10 logits promoted by those features
# top_10_logits_df = pd.DataFrame(
#     [MODEL.to_str_tokens(i) for i in inds[random_indices]],
#     index=random_indices.tolist(),
# ).T
# top_10_logits_df

# # %%

# from datasets import load_dataset

# dataset = load_dataset(DATASET_NAME)["train"]
# dataset.set_format("pt")

# # %%
# import plotly.express as px

# sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

# with torch.no_grad():
#     # activation store can give us tokens.
#     batch_tokens = dataset[:32]["input_ids"].to(device)
#     _, cache = MODEL.run_with_cache(batch_tokens, prepend_bos=False)

#     # Use the SAE
#     feature_acts = sae.encode(cache[sae.cfg.hook_name])
#     sae_out = sae.decode(feature_acts)

#     # save some room
#     del cache

#     # ignore the bos token, get the number of features that activated in each token, averaged accross batch and position
#     l0 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()
#     print("average l0", l0.mean().item())
#     px.histogram(l0.flatten().cpu().numpy()).show()

# # %%

# from transformer_lens import utils
# from functools import partial


# # next we want to do a reconstruction test.
# def reconstr_hook(activation, hook, sae_out):
#     return sae_out

# def zero_abl_hook(activation, hook):
#     return torch.zeros_like(activation)

# print("Orig", MODEL(batch_tokens, return_type="loss").item())
# print(
#     "reconstr",
#     MODEL.run_with_hooks(
#         batch_tokens,
#         fwd_hooks=[
#             (
#                 sae.cfg.hook_name,
#                 partial(reconstr_hook, sae_out=sae_out),
#             )
#         ],
#         return_type="loss",
#     ).item(),
# )
# print(
#     "Zero",
#     MODEL.run_with_hooks(
#         batch_tokens,
#         return_type="loss",
#         fwd_hooks=[(sae.cfg.hook_name, zero_abl_hook)],
#     ).item(),
# )


