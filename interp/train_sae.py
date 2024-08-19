# %%
"""
Configured from
https://colab.research.google.com/github/jbloomAus/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb
"""
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
from transformer_lens import HookedTransformer

from interp import GPT_SMALL
from interp.culture import load_hooked

device = "mps"

total_training_steps = 30_000  # probably we should do more
batch_size = 4096
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

hook_layer = 8
hook_name = f"blocks.{hook_layer}.hook_mlp_out"

DATASET_NAME = "tommyp111/culture-puzzles-1M"

CTX_SIZE = 405

from pathlib import Path

pretrained_path = Path(__file__).parents[1] / "culture-gpt-0-sae"

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    model_name="culture-gpt-0-sae",  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
    hook_layer=hook_layer,
    hook_name=hook_name,
    d_in=GPT_SMALL.d_model,  # width of the mlp output.
    dataset_path=DATASET_NAME,
    is_dataset_tokenized=True,
    streaming=True,  # we could pre-download the token dataset if it was small.
    # SAE Parameters
    mse_loss_normalization=None,  # We won't normalize the mse loss,
    expansion_factor=32,  # the width of the SAE. Larger will result in better stats but slower training.
    b_dec_init_method="zeros",  # The geometric median can be used to initialize the decoder weights.
    apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
    normalize_sae_decoder=False,
    scale_sparsity_penalty_by_decoder_norm=True,
    decoder_heuristic_init=True,
    init_encoder_as_decoder_transpose=True,
    normalize_activations="expected_average_only_in",
    # Training Parameters
    lr=2e-5,  # lower the better, we'll go fairly high to speed up the tutorial.
    adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
    adam_beta2=0.999,
    lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
    lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
    lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
    l1_coefficient=5,  # will control how sparse the feature activations are
    l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
    lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
    train_batch_size_tokens=batch_size,
    context_size=CTX_SIZE,  # will control the lenght of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
    # Activation Store Parameters
    n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
    training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
    store_batch_size_prompts=16,
    # Resampling protocol
    use_ghost_grads=False,  # we don't use ghost grads anymore.
    feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
    dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
    dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
    # WANDB
    log_to_wandb=True,  # always use wandb unless you are just testing code.
    wandb_project="culture_sae",
    wandb_log_frequency=30,
    eval_every_n_wandb_logs=20,
    # Misc
    device=device,
    seed=42,
    n_checkpoints=10,
    checkpoint_path="checkpoints",
    dtype="float32",
    from_pretrained_path=str(pretrained_path.resolve()),
)

model = load_hooked(0)
assert isinstance(model, HookedTransformer)

# if __name__ == "__main__":
#     sparse_autoencoder = SAETrainingRunner(cfg, override_model=model).run()

# %%

from sae_lens import SAE

sae = SAE.load_from_pretrained(str(pretrained_path), device=device)

# %%

import torch

import pandas as pd

# Let's start by getting the top 10 logits for each feature
projection_onto_unembed = sae.W_dec @ model.W_U


# get the top 10 logits.
vals, inds = torch.topk(projection_onto_unembed, 10, dim=1)

# get 10 random features
random_indices = torch.randint(0, projection_onto_unembed.shape[0], (10,))

# Show the top 10 logits promoted by those features
top_10_logits_df = pd.DataFrame(
    [model.to_str_tokens(i) for i in inds[random_indices]],
    index=random_indices.tolist(),
).T
top_10_logits_df

# %%

from datasets import load_dataset

dataset = load_dataset(DATASET_NAME)["train"]
dataset.set_format("pt")

# %%
import plotly.express as px

sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

with torch.no_grad():
    # activation store can give us tokens.
    batch_tokens = dataset[:32]["input_ids"].to(device)
    _, cache = model.run_with_cache(batch_tokens, prepend_bos=False)

    # Use the SAE
    feature_acts = sae.encode(cache[sae.cfg.hook_name])
    sae_out = sae.decode(feature_acts)

    # save some room
    del cache

    # ignore the bos token, get the number of features that activated in each token, averaged accross batch and position
    l0 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()
    print("average l0", l0.mean().item())
    px.histogram(l0.flatten().cpu().numpy()).show()

# %%

from transformer_lens import utils
from functools import partial


# next we want to do a reconstruction test.
def reconstr_hook(activation, hook, sae_out):
    return sae_out

def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)

print("Orig", model(batch_tokens, return_type="loss").item())
print(
    "reconstr",
    model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[
            (
                sae.cfg.hook_name,
                partial(reconstr_hook, sae_out=sae_out),
            )
        ],
        return_type="loss",
    ).item(),
)
print(
    "Zero",
    model.run_with_hooks(
        batch_tokens,
        return_type="loss",
        fwd_hooks=[(sae.cfg.hook_name, zero_abl_hook)],
    ).item(),
)

