import torch as t
import numpy as np

from interp import GPT_SMALL

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

__all__ = ["repr_grid", "create_tokenizer"]


def create_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {f"{i:X}": i for i in range(11)}  # 0-A
    vocab.update({"X": 11, "f_X": 12, "Y": 13, "f_Y": 14, "[UNK]": 15})
    model = WordLevel(vocab, unk_token="[UNK]")
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore

    tokenizer.post_processor = TemplateProcessing(  # type: ignore
        single="0 $A",  # 0 as BOS
        special_tokens=[("0", 0)],  # '0' is both the token and its ID
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="0",
        bos_token="0",
        eos_token="0",
        model_max_length=GPT_SMALL.n_ctx,
    )


def repr_grid(grids) -> str:
    if isinstance(grids, t.Tensor): grids = grids.cpu().numpy()
    elif not isinstance(grids, np.ndarray): grids = np.array(grids)
    if grids.ndim == 0: grids = grids[None]

    if grids[0] == 0: grids = grids[1:] # pop first 0 pad if exists

    # ANSI color codes
    colors = [
        "\033[0m",  # 0: White
        "\033[36m",  # 1: Cyan
        "\033[31m",  # 2: Red
        "\033[38;5;172m",  # 3: Light Yellow
        "\033[33m",  # 4: Yellow
        "\033[92m",  # 5: Light Green
        "\033[32m",  # 6: Green
        "\033[95m",  # 7: Purple
        "\033[34m",  # 8: Blue
        "\033[38;5;208m",  # 9: Orange
        "\033[35m",  # 10: Magenta
    ]

    block = "███"
    grid_labels = {11: "A", 12: "f(A)", 13: "B", 14: "f(B)"}
    repr_string = ""

    for i in range(0, 404, 101):
        if i > 0: repr_string += "\n\n"
        try:
            repr_string += f"{grid_labels[grids[i]]}:\n"
        except IndexError:
            return repr_string

        for j in range(1, 101):
            if j % 10 == 1 and j > 1: repr_string += "\n"
            try: value = grids[i + j]
            except IndexError: return repr_string
            repr_string += f"{colors[value]}{block}\033[0m"
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
