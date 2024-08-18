# %%
from interp.all import *
from datasets import load_dataset, Dataset, DatasetDict
from transformer_lens import HookedTransformer
from sae_lens import SAE
from pathlib import Path
from functools import partial


device = get_device()
sae_pretrained_path = Path(__file__).parents[1] / "culture-gpt-0-sae"

model: HookedTransformer
model = load_hooked(0).eval().to(device)  # type: ignore


sae = SAE.load_from_pretrained(str(sae_pretrained_path), device=str(device))
sae.eval()


dataset = load_dataset("tommyp111/culture-puzzles-1M-partitioned")
assert isinstance(dataset, DatasetDict)
dataset.set_format("pt")

# %%


@t.no_grad()
def quiz_diff_mask(quizzes):
    final_grid_slice = slice(
        -100, None
    )  # not including the special tokens (always different and easy)
    second_final_grid_slice = slice(-201, -101)
    return t.argwhere(
        quizzes[:, final_grid_slice] != quizzes[:, second_final_grid_slice]
    )


def zero_abl_hook(activation, hook):
    return t.zeros_like(activation)


# %%

# batch = dataset["frame"]["input_ids"][:100].to(device) # shave off last token
for task, data in dataset.items():
    print("\ntask:", task)
    batch = data["input_ids"][:100].to(device)
    batch_stripped = batch[:, :-1]
    # mask = quiz_diff_mask(batch_stripped)
    with t.inference_mode():
        model.reset_hooks()
        orig_logits, orig_loss = model(
            batch_stripped, return_type="both", loss_per_token=True
        )

        # orig_tfm_loss = orig_loss[mask].mean().item()
        # orig_tfm_logits = orig_loss[mask].mean().item()
        orig_tfm_loss = orig_loss.mean().item()
        orig_tfm_logits = orig_loss.mean().item()

        print(f"orig loss: {orig_tfm_loss:.2e}")
        for i in range(model.cfg.n_layers):
            logits, loss = model.run_with_hooks(
                batch_stripped,
                return_type="both",
                fwd_hooks=[(f"blocks.{i}.hook_mlp_out", zero_abl_hook)],
                loss_per_token=True,
            )
            # tfm_loss = loss[mask].mean().item()
            # tfm_logits = logits[mask]
            tfm_loss = loss.mean().item()
            tfm_logits = logits
            print(f"Layer {i} diff: {tfm_loss - orig_tfm_loss:.2e}")

# %%


def ablate_mlp_single_task(batch, layer):
    # not including the special tokens (always different and easy)
    final_grid_slice = slice(-100, None)

    batch_stripped = batch[:, :-1]
    diff_mask = quiz_diff_mask(batch_stripped)

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
    model.add_hook(sae.cfg.hook_name, partial(reconstr_hook, sae_out=sae_out[i])) # type: ignore
    correct, _ = generate_and_print(model, batch[wrong_because_ablate[i]], temperature=0.0)
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
