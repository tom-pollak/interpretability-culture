# %%
### Imports ###

import sys, time, tqdm, math
from dataclasses import dataclass
from pathlib import Path

import torch as t
import torch.nn.functional as F


root_dir = Path(__file__).parents[1]
interp_dir = root_dir / "interp"
interp_result_dir = interp_dir / "results_interp"

culture_dir = root_dir / "culture"
result_dir = culture_dir / "results_noise"
assert culture_dir.exists() and result_dir.exists()

if str(culture_dir) not in sys.path:
    sys.path.append(str(culture_dir.resolve()))

import grids, quiz_machine
import mygpt

# %%

### Setup ###

log_file = open(interp_result_dir / "eval.log", "a")


def log_string(s):
    t = time.strftime("%Y%m%d-%H:%M:%S ", time.localtime())

    if log_file is not None:
        log_file.write(t + s + "\n")
        log_file.flush()

    print(t + s)
    sys.stdout.flush()


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


@dataclass
class GptCfg:
    name: str = "37M"
    dim_model: int = 512
    dim_keys: int = 64
    dim_hidden: int = 2048
    nb_heads: int = 8
    nb_blocks: int = 12


gpt_cfg = GptCfg()

# %%

### Problem Setup ###


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
    result_dir=interp_result_dir,
    prompt_noise=prompt_noise,
    logger=log_string,
    device=main_device,
)


log_string(f"main_device {main_device} gpus {[ str(g) for g in gpus]}")

vocabulary_size = quiz_machine.vocabulary_size()

log_string(f"vocabulary_size {vocabulary_size}")

# %%


def model_transformer_hot(model):
    model.temperature = temperature_hot


def model_transformer_cold(model):
    model.temperature = temperature_cold


# TODO: not sure what this is
# maybe which ones to complete (aka
# 1. Write f(B) -- be creative
# 2. Given f(B), write f(A) A B
# 3. Given A f(A) B write f(B)
#  or
# 1. f(B) given A & f(A) as an example and B -- be creative
#
# > A > f(A) > B ; > f(B)
# < f(B) ; < B < f(A) < A
c_quizzes_procedure = [
    (("f_B", "f_A", "A", "B"), (1, 0, 0, 0), model_transformer_hot),
    (("f_B", "f_A", "A", "B"), (0, 1, 1, 1), model_transformer_cold),
    (("A", "f_A", "B", "f_B"), (0, 0, 0, 1), model_transformer_cold),
    # (("f_B", "f_A", "A", "B"), (0, 0, 1, 1), model_transformer_cold),
    # (("A", "f_A", "B", "f_B"), (0, 0, 0, 1), model_transformer_cold),
]

# %%

### Model Setup with Quizzes ###

nb_gpts = 4
models = []

for k in range(nb_gpts):
    log_string(f"creating model {k} and its w_quizzes")

    model = mygpt.MyGPT(
        vocabulary_size=vocabulary_size,
        dim_model=gpt_cfg.dim_model,
        dim_keys=gpt_cfg.dim_keys,
        dim_hidden=gpt_cfg.dim_hidden,
        nb_heads=gpt_cfg.nb_heads,
        nb_blocks=gpt_cfg.nb_blocks,
        causal=True,
        dropout=dropout,
    ).to(main_device)

    model.main_test_accuracy = 0.0  # type: ignore
    model.id = k  # type: ignore

    model.test_w_quizzes = quiz_machine.problem.generate_w_quizzes(nb_test_samples)  # type: ignore

    models.append(model)

# %%

### Checkpoint Loading ###

current_epoch = 0
n_epoch = current_epoch + 1

for model in models:
    filename = f"gpt_{model.id:03d}.pth"

    try:
        d = t.load(result_dir / filename, map_location="cpu")
        model.load_state_dict(d[0])
        model.main_test_accuracy = d[1]
        log_string(f"successfully loaded {filename}")
    except FileNotFoundError:
        log_string(f"cannot find {filename}")
        pass

filename = "c_quizzes.pth"
try:
    quiz_machine.load_c_quizzes(result_dir / filename)
    log_string(f"successfully loaded {filename}")
except FileNotFoundError:
    log_string(f"cannot find {filename}")
    pass

filename = "state.pth"
try:
    state = t.load(result_dir / filename, weights_only=False)
    log_string(f"successfully loaded {filename}")
    current_epoch = state["current_epoch"]
except FileNotFoundError:
    log_string(f"cannot find {filename}")

nb_parameters = sum(p.numel() for p in models[0].parameters())
log_string(f"nb_parameters {nb_parameters} ({int(nb_parameters/1e6)}M)")

cta = " ".join([f"{float(m.main_test_accuracy):.04f}" for m in models])
log_string(f"current_test_accuracies {cta}")

# %%

def run_tests(model, quiz_machine, local_device=main_device):
    with t.autograd.no_grad():
        model.eval().to(local_device)

        nb_test_samples, acc_test_loss = 0, 0.0
        nb_samples_accumulated = 0


        for input in tqdm.tqdm(src, dynamic_ncols=True, desc="test"):
            input = input.to(local_device)
            output = model(mygpt.BracketedSequence(input)).x
            loss = F.cross_entropy(output.transpose(1, 2), input)
            acc_test_loss += loss.item() * input.size(0)
            nb_test_samples += input.size(0)

        test_perplexity = math.exp(min(100, acc_test_loss / nb_test_samples))

        log_string(f"test_perplexity {n_epoch} model {model.id} {test_perplexity}")

        model.main_test_accuracy = quiz_machine.produce_results(
            n_epoch=n_epoch,
            model=model,
            input=full_input[:2000],
            result_dir=interp_result_dir,
        )



# %%

model = models[0]
full_input, _ = quiz_machine.data_input(model, split="test")
src = full_input.split(batch_size)[:3] # 3 batches

# %%

with t.inference_mode():
    input = src[0].to(main_device)
    output = model(mygpt.BracketedSequence(input)).x
    print(input.shape, output.shape)
    loss = F.cross_entropy(output.transpose(1, 2), input)
    print(loss)

# %%

sequence = t.tensor([
    [1, 5, 6, 7, 2, 8, 9, 10, 3, 11, 12, 13, 4, 0, 0],  # Example 1
    [1, 5, 5, 5, 2, 6, 6, 6, 3, 7, 7, 7, 4, 0, 0],  # Example 2
])

bs = mygpt.BracketedSequence(sequence, first=0, nb=8)
# %%

from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoTokenizer

# %%

# even are attention, odd are mlp

layers = [] # 24 in total

model.trunk[1].f
