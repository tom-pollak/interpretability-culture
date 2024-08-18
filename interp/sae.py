# %%
from interp.all import *
from datasets import load_dataset, Dataset, DatasetDict
from transformer_lens import HookedTransformer
from sae_lens import SAE
from pathlib import Path


device = get_device()
sae_pretrained_path = Path(__file__).parents[1] / "culture-gpt-0-sae"

model = load_hooked(0)
assert isinstance(model, HookedTransformer)
model = model.eval().to(device)


sae = SAE.load_from_pretrained(str(sae_pretrained_path), device=str(device))


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
for label, data in dataset.items():
    print("task:", label)
    batch = data["input_ids"][:100].to(device)
    batch_stripped = batch[:, :-1]
    # mask = quiz_diff_mask(batch_stripped)
    with t.inference_mode():
        model.reset_hooks()
        orig_logits, orig_loss = model(batch_stripped, return_type="both", loss_per_token=True)

        # orig_tfm_loss = orig_loss[mask].mean().item()
        # orig_tfm_logits = orig_loss[mask].mean().item()
        orig_tfm_loss = orig_loss.mean().item()
        orig_tfm_logits = orig_loss.mean().item()

        print(f"orig loss: {orig_tfm_loss:.2e}")
        for i in [0, 8]: # range(model.cfg.n_layers):
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
orig_correct, orig_pred = generate(model, batch[:100])

model.add_hook("blocks.8.hook_mlp_out", zero_abl_hook)  # type: ignore
abl_correct, abl_pred = generate(model, batch[:100], verbose=False, prefix_0=False)
model.reset_hooks()


# %%
model.add_hook("blocks.8.hook_mlp_out", zero_abl_hook)  # type: ignore
generate_and_print(model, batch[:1], prefix_0=False)
model.reset_hooks()
# %%

"""
Using train_sae.py, trained a model (somewhat arbitarily)

"""
