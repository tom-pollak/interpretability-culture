import torch as t
import torch.nn as nn
import numpy as np

from interp import GPT_SMALL

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

__all__ = [
    "TOK_PREPROCESS",
    "prep_quiz",
    "sinusoidal_positional_encoding",
    "repr_grid",
    "create_tokenizer"
]


TOK_PREPROCESS = nn.ConstantPad1d((1, -1), value=0)  # pads with a 0 start token, shaves off last token


def prep_quiz(quizzes, prefix_0=True, slice_at=305):
    if prefix_0:
        # different from TOK_PREPROCESS, this doesn't shave off the last token
        quizzes = t.cat((
            t.zeros(quizzes.shape[0], 1, device=quizzes.device, dtype=quizzes.dtype),
            quizzes
        ), dim=1)
    if slice_at is not None:
        quizzes = quizzes[:, :slice_at]
    return quizzes


def sinusoidal_positional_encoding(
    seq_length: int, emb_dim: int, max_length: float = 1e5
):  # (seq_length, emb_dim)
    T = t.arange(seq_length)[:, None]
    J = t.arange(emb_dim)[None, :]
    K = J % 2
    pe = t.sin(T / (max_length ** ((J - K) / emb_dim)) + t.pi / 2 * K)
    return pe


def create_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {f"{i:x}": i for i in range(11)}  # 0-A
    vocab.update({"X": 11, "f_X": 12, "Y": 13, "f_Y": 14, "[UNK]": 15})
    model = WordLevel(vocab, unk_token="[UNK]")
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="0",
        bos_token="0",
        model_max_length=GPT_SMALL.n_ctx + 1,
    )


def repr_grid(grids) -> str:
    # ANSI color codes
    colors = [
        "0m",   # 0: White
        "36m",  # 1: Cyan
        "31m",  # 2: Red
        "38;5;219m", # 3: Light Pink
        "33m",  # 4: Yellow
        "38;5;87m",  # 5: Light Turquoise
        "32m",  # 6: Green
        "38;5;106m",  # 7: Olive Green
        "34m",  # 8: Blue
        "38;5;208m",  # 9: Orange
        "35m",  # 10: Magenta
    ]
    block = "███"
    grid_labels = {11: "X", 12: "f(X)", 13: "Y", 14: "f(Y)"}

    def _fmt_block(value: int) -> str:
        if value in grid_labels:
            return f"{grid_labels[value]}"
        return f"\033[{colors[value]}{block}\033[0m"

    if isinstance(grids, t.Tensor): grids = grids.cpu().numpy()
    elif not isinstance(grids, np.ndarray): grids = np.array(grids)
    if grids.ndim == 0: grids = grids[None]
    if grids[0] == 0: grids = grids[1:] # pop first 0 pad if exists

    repr_string = ""
    for i in range(0, 404, 101):
        if i > 0: repr_string += "\n\n"
        for j in range(101):
            if j % 10 == 1: repr_string += "\n"
            try:
                repr_string += _fmt_block(grids[i + j])
            except IndexError:
                return repr_string
    return repr_string


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a tokenizer for the Culture model"
    )
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    from interp import *
    from interp.culture import *

    device = get_device()

    ht = load_hooked(0)
    qm = load_quizzes()
    model = ht.eval().to(device)  # type: ignore
    gen_test_w_quiz_(model, qm, n=100)
    dataset, from_w = qm.data_input(model, split="test")

    row = dataset[0]
    print("gpt generated:", not from_w[0].item())
    print(repr_grid(row))
    print("\n\n###############\n\n")

    slice_length = 75
    row_slice = row[:slice_length]
    tokenizer = create_tokenizer()
    decoded = tokenizer.decode(row_slice)
    encoded = tokenizer.encode(decoded, add_special_tokens=False) # don't add first 0
    print("Reencoded slice:\n")
    print(repr_grid(encoded))

    if args.push_to_hub:
        print("\n\npushing to hub... ", end="")
        tokenizer.push_to_hub("tommyp111/culture-grid-tokenizer")
        print("done.")
