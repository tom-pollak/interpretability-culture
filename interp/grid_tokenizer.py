import numpy as np
from dataclasses import dataclass

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


@dataclass
class GridTokenizerConfig:
    unk_token: str = "[UNK]"
    pad_token: str = "0"
    bos_token: str = "0"
    eos_token: str = "0"
    model_max_length: int = 404  # 14 normal, 0, UNK


class GridTokenizerFast(PreTrainedTokenizerFast):
    def __init__(self, tokenizer=None, **kwargs):
        if tokenizer is None:
            # Create the base vocabulary (1-E for 14 tokens, plus 0 as special and [UNK])
            vocab = {f"{i:X}": i for i in range(15)}  # 1-F
            vocab.update({"[UNK]": 15})
            model = WordLevel(vocab, unk_token="[UNK]")
            tokenizer = Tokenizer(model)
            tokenizer.pre_tokenizer = Whitespace()  # type: ignore

        super().__init__(tokenizer_object=tokenizer, **kwargs)

    def _decode_single_token(self, token, skip_special_tokens=False, **kwargs):
        return super().decode(
            [token], skip_special_tokens=skip_special_tokens, **kwargs
        )

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=None,
        **kwargs,
    ):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if isinstance(token_ids, list):
            token_ids = np.array(token_ids)

        if token_ids.ndim == 0:
            token_ids = token_ids[None]

        if len(token_ids) != 404:
            return super().decode(
                token_ids, skip_special_tokens=skip_special_tokens, **kwargs
            )

        grids = []
        for i in range(4):
            start = i * 101
            grid = token_ids[start + 1 : start + 101].reshape(10, 10)
            grids.append(grid)

        decoded = ""
        for i, grid in enumerate(grids):
            decoded += f"Grid {i+1} (Start token: {self._decode_single_token(token_ids[i*101], skip_special_tokens=skip_special_tokens, **kwargs)}):\n"
            for row in grid:
                decoded += (
                    " ".join(
                        [
                            self._decode_single_token(
                                token, skip_special_tokens=skip_special_tokens, **kwargs
                            )
                            for token in row
                        ]
                    )
                    + "\n"
                )
            decoded += "\n"

        return decoded
