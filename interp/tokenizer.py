from interp import GPT_SMALL

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import numpy as np

# Create the base vocabulary (1-F for 15 tokens, plus 0 as special and [UNK])
vocab = {f"{i:X}": i for i in range(1, 16)}  # 1-F
vocab.update({"0": 0, "[UNK]": 16})
model = WordLevel(vocab, unk_token="[UNK]")
tokenizer = Tokenizer(model)
tokenizer.pre_tokenizer = Whitespace() # type: ignore


class GridTokenizerFast(PreTrainedTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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


if __name__ == "__main__":
    grid_tokenizer = GridTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="0",
        bos_token="0",
        eos_token="0",
        model_max_length=GPT_SMALL.n_ctx,  # 4 * (1 + 10 * 10)
    )

    # Test the tokenizer

    base_tokens = [f"{(i%14)+1:X}" for i in range(400)]
    for i in range(0, 400, 101):
        base_tokens.insert(i, '0')
    test_input = " ".join(base_tokens)

    encoded = grid_tokenizer.encode(test_input, add_special_tokens=False)
    decoded = grid_tokenizer.decode(encoded)
    print("Encoded (first 20 tokens):", encoded[:20])
    print("\nDecoded:")
    print(decoded)

    # grid_tokenizer.push_to_hub()
