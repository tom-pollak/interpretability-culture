from interp import (
    set_seed,
    get_device,
    num_params,
    RESULT_DIR,
    INTERP_RESULTS_DIR,
    GPT_SMALL,
    QuizMachineConfig,
    QM_CONFIG,
)

import math
import tqdm
from typing import Optional

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from transformer_lens import HookedTransformer, HookedTransformerConfig

import mygpt
import grids
from quiz_machine import QuizMachine

__all__ = [
    "TOK_PREPROCESS",
    "sinusoidal_positional_encoding",
    "load_culture",
    "load_hooked",
    "add_preproc",
    "load_quizzes",
    "gen_test_w_quiz_",
    "run_tests",
    "convert_culture_weights",
]


TOK_PREPROCESS = nn.ConstantPad1d((1, -1), value=0)  # pads with a 0 start token


def sinusoidal_positional_encoding(
    seq_length: int, emb_dim: int, max_length: float = 1e5
):  # (seq_length, emb_dim)
    T = t.arange(seq_length)[:, None]
    J = t.arange(emb_dim)[None, :]
    K = J % 2
    pe = t.sin(T / (max_length ** ((J - K) / emb_dim)) + t.pi / 2 * K)
    return pe


# ----- loading models -----


def load_culture() -> list[mygpt.MyGPT]:
    def _update_state_dict_keys(sd):
        "Since I added ActivationCache to MyGPT, I need to update state_dict keys to load properly"
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("trunk."):
                parts = k.split(".")
                new_key = f"{parts[0]}.{parts[1]}.f.{'.'.join(parts[2:])}"
            else:
                parts = k.split(".")
                new_key = f"{parts[0]}.f.{'.'.join(parts[1:])}"
            new_sd[new_key] = v
        return new_sd

    nb_gpts = 4
    models = []
    for k in range(nb_gpts):
        model = mygpt.MyGPT(
            vocabulary_size=GPT_SMALL.d_vocab,
            dim_model=GPT_SMALL.d_model,
            dim_keys=GPT_SMALL.d_head,
            dim_hidden=GPT_SMALL.d_mlp,
            nb_heads=GPT_SMALL.n_heads,
            nb_blocks=GPT_SMALL.n_layers,
            causal=True,
            dropout=0.0,
            activation_cache=False,
        )

        d = t.load(
            RESULT_DIR / f"gpt_{k:03d}.pth", map_location="cpu", weights_only=True
        )

        sd = _update_state_dict_keys(d[0])
        model.load_state_dict(sd)

        model.id = k  # type: ignore
        model.main_test_accuracy = d[1]

        print(f"loaded model {k}")
        models.append(model)
    return models


def load_hooked() -> list[HookedTransformer]:
    models = load_culture()

    # convert weights
    state_dicts = [convert_culture_weights(model, GPT_SMALL) for model in models]

    # load
    hooked_tranformers = []
    for k, state_dict in enumerate(state_dicts):
        ht = HookedTransformer(GPT_SMALL)
        ht.load_and_process_state_dict(
            state_dict,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
            refactor_factored_attn_matrices=False,
        )
        ht.id = k  # type: ignore
        hooked_tranformers.append(ht)
    return hooked_tranformers


def add_preproc(models: list[HookedTransformer]) -> list[nn.Sequential]:
    "for run_tests"
    preproc_models = []
    for model in models:
        preproc_model = nn.Sequential(TOK_PREPROCESS, model)
        preproc_model.id = model.id
        preproc_models.append(preproc_model)
    return preproc_models


# ----- loading datasets -----


def load_quizzes(cfg: Optional[QuizMachineConfig] = None, logger=print, device=None):
    cfg = cfg or QM_CONFIG
    device = device or get_device()

    grids_world_tasks = (
        "replace_color,translate,grow,half_fill,frame,detect,corners,contact"
    )

    problem = grids.Grids(
        max_nb_cached_chunks=cfg.nb_test_samples // 100,
        chunk_size=100,
        nb_threads=1,
        tasks=grids_world_tasks,
    )

    qm = QuizMachine(
        problem=problem,
        batch_size=cfg.inference_batch_size,
        result_dir=INTERP_RESULTS_DIR,
        prompt_noise=cfg.prompt_noise,
        logger=logger,
        use_brack_seq=False,
        device=device,
    )
    qm.load_c_quizzes(RESULT_DIR / "c_quizzes.pth")
    return qm


def gen_test_w_quiz_(models, qm: QuizMachine, n: int = 2000):
    if not isinstance(models, list):
        models = [models]

    for model in models:
        model.test_w_quizzes = qm.problem.generate_w_quizzes(n)


# ----- eval -----


@t.inference_mode()
def run_tests(model, quiz_machine: QuizMachine, device=None):
    device = device or get_device()
    model.eval().to(device)

    nb_test_samples, acc_test_loss = 0, 0.0

    full_input, _ = quiz_machine.data_input(model, split="test")
    src = full_input.split(QM_CONFIG.inference_batch_size)

    for input in tqdm.tqdm(src, dynamic_ncols=True, desc="test"):
        input = input.to(device)
        output = model(input)
        loss = F.cross_entropy(output.transpose(1, 2), input)
        acc_test_loss += loss.item() * input.size(0)
        nb_test_samples += input.size(0)

    test_perplexity = math.exp(min(100, acc_test_loss / nb_test_samples))

    quiz_machine.logger(f"test_perplexity ({model.id}) {test_perplexity:.04f}")

    model.main_test_accuracy = quiz_machine.produce_results(
        n_epoch=10_000,
        model=model,
        input=full_input[:2000],
        result_dir=INTERP_RESULTS_DIR,
    )
    quiz_machine.logger(f"accuracy: {float(model.main_test_accuracy):.04f}")


# ----- convert weights -----


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


######################################################################

if __name__ == "__main__":
    import einops
    import argparse

    # used for eval accuracy
    parser = argparse.ArgumentParser(
        description="Convert culture weights to hooked transformer"
    )
    parser.add_argument("num_test_samples", type=int, default=200, help="(1-2000)")
    args = parser.parse_args()
    if args.num_test_samples < 1 or args.num_test_samples > 2000:
        parser.error("num_test_samples must be between 1 and 2000")

    # test positional encoding
    mygpt_pos_enc = mygpt.AddPositionalEncoding(1e5)(
        mygpt.BracketedSequence(t.zeros(32, 404, 512))
    ).x
    pos_enc = einops.repeat(
        sinusoidal_positional_encoding(404, 512, 1e5),
        "pos d_model -> batch pos d_model",
        batch=32,
    )
    print("positional encoding correct:", t.allclose(mygpt_pos_enc, pos_enc))

    device = get_device()

    # load culture
    culture_models = load_culture()

    nparams = num_params(culture_models[0])
    print(f"nb_parameters {nparams} ({int(nparams/1e6)}M)")

    cta = " ".join([f"{float(m.main_test_accuracy):.04f}" for m in culture_models])
    print(f"current_test_accuracies {cta}")

    # load hooked
    hooked_transformers = add_preproc(load_hooked())
    print(hooked_transformers[0][1])

    # dataset load
    qm = load_quizzes()

    # eval logits same on one batch
    cult_m0 = culture_models[0].eval().to(device)
    ht_m0 = hooked_transformers[0].eval().to(device)

    set_seed()
    gen_test_w_quiz_(ht_m0, qm, n=QM_CONFIG.batch_size)
    inp = qm.data_input(ht_m0, split="test")[0].to(device)
    with t.inference_mode():
        cult_logits = cult_m0(mygpt.BracketedSequence(inp)).x
        ht_logits = ht_m0(inp)
    print(
        "MyGPT & HookedTransformer allclose:",
        t.allclose(cult_logits, ht_logits, atol=5e-4),
    )

    # eval accuracy
    print(f"evaluating accuracy ({args.num_test_samples} samples)")

    for model in hooked_transformers:
        print(f"--- model {model.id} ---")
        set_seed()
        gen_test_w_quiz_(model, qm, n=args.num_test_samples)
        run_tests(model, qm)

    hta = " ".join([f"{float(m.main_test_accuracy):.04f}" for m in hooked_transformers])
    print(f"hooked transformer accuracies {hta}")