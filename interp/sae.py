# %%
# %%


# %%

"""
"""

# %%
"""
"""

# %%
"""
"""

# %%


"""
"""

# %%

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


