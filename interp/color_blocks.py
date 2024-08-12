import torch as t
import numpy as np


def tensor_to_color_blocks(grids):
    # Ensure the tensor is on CPU and convert to a numpy array
    if isinstance(grids, t.Tensor):
        grids = grids.cpu().numpy()
    elif not isinstance(grids, list):
        np.array(grids)

    # ANSI color codes
    colors = [
        "\033[0m",   # 0: White
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
    grid_labels = {11: "0", 12: "1", 13: "2", 14: "3"}
    repr_string = ""

    for i in range(0, 404, 101):
        # Add a separator between grids
        if i > 0:
            repr_string += "\n\n"

        # Convert the grid token
        repr_string += f"{grid_labels[grids[i]]}:\n"

        # Convert the 10x10 grid
        for j in range(1, 101):
            if j % 10 == 1 and j > 1:
                repr_string += "\n"
            value = grids[i + j]
            repr_string += f"{colors[value]}{block}\033[0m"

    return repr_string


# Example usage:
# tensor = torch.randint(0, 15, (404,))
# print(tensor_to_color_blocks(tensor))



from interp import *
from interp.culture import *


device = get_device()

hts = load_hooked()
qm = load_quizzes()
model = hts[0].eval().to(device)  # type: ignore
gen_test_w_quiz_(model, qm, n=100)
dataset, from_w = qm.data_input(model, split="test")
print(from_w[1])
print(tensor_to_color_blocks(dataset[1]))

# %%
