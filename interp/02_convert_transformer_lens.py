# %%
### Imports ###

import sys, time, tqdm, math
from dataclasses import dataclass, asdict
from pathlib import Path

import torch as t
from torch import nn
import torch.nn.functional as F
from einops import einops

from transformer_lens import HookedTransformer, HookedTransformerConfig

from interp import INTERP_DIR, INTERP_RESULTS_DIR, RESULT_DIR


import mygpt

# %%

### Setup ###

seed = 307451
t.manual_seed(seed)
if t.cuda.is_available():
    t.cuda.manual_seed_all(seed)

# %%

### Hyperparameters ###

if t.cuda.is_available():
    gpus = [t.device(f"cuda:{n}") for n in range(t.cuda.device_count())]
elif t.backends.mps.is_available():
    gpus = [t.device("mps")]
else:
    gpus = [t.device("cpu")]

main_device = gpus[0]


# 37M
cfg = HookedTransformerConfig(
    d_model=512,
    d_head=64,
    d_mlp=2048,
    n_heads=8,
    n_layers=12,
    n_ctx=404,
    d_vocab=15,
    act_fn="relu",
    final_ln=False,
)


nb_gpts = 4
models = []
vocabulary_size = 15

for k in range(nb_gpts):
    model = mygpt.MyGPT(
        vocabulary_size=vocabulary_size,
        dim_model=cfg.d_model,
        dim_keys=cfg.d_head,
        dim_hidden=cfg.d_mlp,
        nb_heads=cfg.n_heads,
        nb_blocks=cfg.n_layers,
        causal=True,
        dropout=0.0,
        activation_cache=False,
    ).to(main_device)

    model.id = k  # type: ignore
    models.append(model)

# %%

### Checkpoint Loading ###

current_epoch = 0
n_epoch = current_epoch + 1


def update_state_dict_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("trunk."):
            parts = key.split(".")
            new_key = f"{parts[0]}.{parts[1]}.f.{'.'.join(parts[2:])}"
        else:
            parts = key.split(".")
            new_key = f"{parts[0]}.f.{'.'.join(parts[1:])}"
        new_state_dict[new_key] = value
    return new_state_dict

for model in models:
    filename = f"gpt_{model.id:03d}.pth"

    try:
        d = t.load(RESULT_DIR / filename, map_location="cpu")
        sd = update_state_dict_keys(d[0])
        model.load_state_dict(sd)
        model.main_test_accuracy = d[1]
        print(f"successfully loaded {filename}")
    except FileNotFoundError:
        print(f"cannot find {filename}")
        pass


filename = "state.pth"
try:
    state = t.load(RESULT_DIR / filename, weights_only=False)
    print(f"successfully loaded {filename}")
    current_epoch = state["current_epoch"]
except FileNotFoundError:
    print(f"cannot find {filename}")

nb_parameters = sum(p.numel() for p in models[0].parameters())
print(f"nb_parameters {nb_parameters} ({int(nb_parameters/1e6)}M)")

cta = " ".join([f"{float(m.main_test_accuracy):.04f}" for m in models])
print(f"current_test_accuracies {cta}")

# %%

def sinusoidal_positional_encoding(
    seq_length: int, emb_dim: int, max_length: float = 1e5
):  # (seq_length, emb_dim)
    T = t.arange(seq_length)[:, None]
    J = t.arange(emb_dim)[None, :]
    K = J % 2
    pe = t.sin(T / (max_length ** ((J - K) / emb_dim)) + math.pi / 2 * K)
    return pe


mygpt_pos_enc = mygpt.AddPositionalEncoding(1e5)(mygpt.BracketedSequence(t.zeros(32, 404, 512))).x
pos_enc = einops.repeat(sinusoidal_positional_encoding(404, 512, 1e5), "pos d_model -> batch pos d_model", batch=32)

print("positional encoding correct:", t.allclose(mygpt_pos_enc, pos_enc))

# %%

"""
state_dict: {
# embed
embed.W_E (d_vocab, d_model)

pos_embed.W_pos (n_ctx, d_model) # ooo this might be tricky sinusodal is not currently supported

# for l in n_layers
    # layernorm 1
    blocks.{l}.ln1.w (d_model)
    blocks.{l}.ln1.b (d_model)

    # attention
    blocks.{l}.attn.W_Q (n_heads, d_model, d_head)
    blocks.{l}.attn.W_K (n_heads, d_model, d_head)
    blocks.{l}.attn.W_V (n_heads, d_model, d_head)

    blocks.{l}.attn.W_O (n_heads, d_head, d_model)

    # ZEROS
    blocks.{l}.attn.b_Q (n_heads, d_head)
    blocks.{l}.attn.b_K (n_heads, d_head)
    blocks.{l}.attn.b_V (n_heads, d_head)

    blocks.{l}.attn.b_O (d_model)


    # layernorm 2
    blocks.{l}.ln2.w (d_model)
    blocks.{l}.ln2.b (d_model)

    # MLP
    blocks.{l}.mlp.W_in (d_model, d_mlp)
    blocks.{l}.mlp.b_in (d_mlp)
    blocks.{l}.mlp.W_out (d_mlp, d_model)
    blocks.{l}.mlp.b_out (d_model)

# unembed
unembed.W_U (d_model, d_vocab)
unembed.b_U (d_vocab)

# final layernorm ZEROS
ln_final.w (d_model)
ln_final.b (d_model)
}
"""

def convert_culture_weights(model, cfg: HookedTransformerConfig) -> dict:
    state_dict = {}
    tok_embedding = model.embedding.f[0].f[0]
    W_E = tok_embedding.weight
    state_dict["embed.W_E"] = W_E

    W_pos = sinusoidal_positional_encoding(seq_length=cfg.n_ctx, emb_dim=cfg.d_model)
    state_dict["pos_embed.W_pos"] = W_pos

    for l in range(cfg.n_layers):
        attn_idx = l * 2  # evens
        mlp_idx = l * 2 + 1  # odds

        ln1 = model.trunk[attn_idx].f.f[0].f[0]
        attn_block = model.trunk[attn_idx].f.f[1]

        ln2 = model.trunk[mlp_idx].f.f.f[0]
        up_proj = model.trunk[mlp_idx].f.f.f[2]
        down_proj = model.trunk[mlp_idx].f.f.f[4]

        W_Q = attn_block.w_q.data.transpose(-1, -2)
        W_K = attn_block.w_k.data.transpose(-1, -2)
        W_V = attn_block.w_v.data.transpose(-1, -2)
        # (dim_v * nb_heads, dim_in)
        # W_O = (
        #     attn_block.w_o.transpose(0, 1)
        #     .contiguous()
        #     .view(cfg.n_heads, cfg.d_head, cfg.d_model)
        # )
        W_O = attn_block.w_o.contiguous().view(cfg.n_heads, cfg.d_head, cfg.d_model)


        state_dict[f"blocks.{l}.ln1.w"] = ln1.weight
        state_dict[f"blocks.{l}.ln1.b"] = ln1.bias

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        # ZEROS
        state_dict[f"blocks.{l}.attn.b_Q"] = t.zeros(cfg.n_heads, cfg.d_head)
        state_dict[f"blocks.{l}.attn.b_K"] = t.zeros(cfg.n_heads, cfg.d_head)
        state_dict[f"blocks.{l}.attn.b_V"] = t.zeros(cfg.n_heads, cfg.d_head)
        state_dict[f"blocks.{l}.attn.b_O"] = t.zeros(cfg.d_model)

        state_dict[f"blocks.{l}.ln2.w"] = ln2.weight
        state_dict[f"blocks.{l}.ln2.b"] = ln2.bias

        state_dict[f"blocks.{l}.mlp.W_in"] = up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = up_proj.bias

        state_dict[f"blocks.{l}.mlp.W_out"] = down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = down_proj.bias

    unembed = model.readout.f.f
    W_U = unembed.weight.T
    b_U = unembed.bias
    state_dict["unembed.W_U"] = W_U
    state_dict["unembed.b_U"] = b_U

    return state_dict


model = models[0]
# print(model)
state_dict = convert_culture_weights(model, cfg)
hooked_tranformer = HookedTransformer(cfg)

preprocess = nn.ConstantPad1d((1, -1), value=0)

# gpt.load_state_dict(state_dict)
errors = hooked_tranformer.load_and_process_state_dict(
    state_dict,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    fold_value_biases=False,
    refactor_factored_attn_matrices=False,
)
print(hooked_tranformer)

gpt = nn.Sequential(preprocess, hooked_tranformer)
gpt.id = 0

# %%

import grids, quiz_machine, mygpt

if t.cuda.is_available():
    gpus = [t.device(f"cuda:{n}") for n in range(t.cuda.device_count())]
elif t.backends.mps.is_available():
    gpus = [t.device("mps")]
else:
    gpus = [t.device("cpu")]

main_device = gpus[0]

batch_size = 25
inference_batch_size = 50

prompt_noise = 0.05
dropout = 0.1

dirty_debug = False
if dirty_debug:
    nb_test_samples = 100
else:
    nb_test_samples = 2000

temperature_hot = 1.5
temperature_cold = 1.0


grids_world_tasks = (
    "replace_color,translate,grow,half_fill,frame,detect,corners,contact"
)

problem = grids.Grids(
    max_nb_cached_chunks=len(gpus) * nb_test_samples // 100,
    chunk_size=100,
    nb_threads=1,
    tasks=grids_world_tasks,
)

quiz_machine = quiz_machine.QuizMachine(
    problem=problem,
    batch_size=inference_batch_size,
    result_dir=INTERP_RESULTS_DIR,
    prompt_noise=prompt_noise,
    logger=print,
    use_brack_seq=False,
    device=main_device,
)

gpt.test_w_quizzes = quiz_machine.problem.generate_w_quizzes(
    nb_test_samples
)  # nb_test_samples  # type: ignore

vocabulary_size = quiz_machine.vocabulary_size()

filename = "c_quizzes.pth"
quiz_machine.load_c_quizzes(RESULT_DIR / filename)
print(f"successfully loaded {filename}")

filename = "state.pth"
state = t.load(RESULT_DIR / filename, weights_only=False)
print(f"successfully loaded {filename}")
current_epoch = state["current_epoch"]



# %%

gpt.eval().to(main_device)
model.eval().to(main_device)

full_input, _ = quiz_machine.data_input(gpt, split="test")
inp = full_input.split(batch_size)[0].to(main_device)


with t.inference_mode():
    logits = gpt(inp)
    mygpt_logits = model(mygpt.BracketedSequence(inp)).x

print("allclose", t.allclose(logits, mygpt_logits, atol=6e-4))

# %%

def run_tests(model, quiz_machine, local_device=main_device):
    with t.autograd.no_grad():
        model.eval().to(local_device)

        nb_test_samples, acc_test_loss = 0, 0.0
        nb_samples_accumulated = 0

        full_input, _ = quiz_machine.data_input(model, split="test")
        src = full_input.split(batch_size)

        for input in tqdm.tqdm(src, dynamic_ncols=True, desc="test"):
            input = input.to(local_device)
            output = model(input)
            loss = F.cross_entropy(output.transpose(1, 2), input)
            acc_test_loss += loss.item() * input.size(0)
            nb_test_samples += input.size(0)

        test_perplexity = math.exp(min(100, acc_test_loss / nb_test_samples))

        print(f"test_perplexity {n_epoch} model {model.id} {test_perplexity}")

        model.main_test_accuracy = quiz_machine.produce_results(
            n_epoch=n_epoch,
            model=model,
            input=full_input[:2000],
            result_dir=INTERP_RESULTS_DIR,
        )


run_tests(gpt, quiz_machine)
print(f"accuracy: {float(gpt.main_test_accuracy):.04f}")

# %%
