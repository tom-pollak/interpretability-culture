from dataclasses import asdict
from interp.grid_tokenizer import GridTokenizerFast, GridTokenizerConfig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a tokenizer for the Culture model"
    )
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    grid_tokenizer = GridTokenizerFast(**asdict(GridTokenizerConfig()))

    # Test the tokenizer

    base_tokens = [f"{(i%14)+1:X}" for i in range(400)]
    for i in range(0, 400, 101):
        base_tokens.insert(i, "0")
    test_input = " ".join(base_tokens)

    encoded = grid_tokenizer.encode(test_input, add_special_tokens=False)
    decoded = grid_tokenizer.decode(encoded)
    print("Encoded (first 20 tokens):", encoded[:20])
    print("\nDecoded:")
    print(decoded)

    if args.push_to_hub:
        print("pushing to hub... ", end="")
        GridTokenizerFast.register_for_auto_class("AutoTokenizer")
        grid_tokenizer.push_to_hub("tommyp111/culture-grid-tokenizer")
        print("done.")
