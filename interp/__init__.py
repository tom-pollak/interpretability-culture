## Exports ##

__all__ = [
    "ROOT_DIR",
    "LOG_DIR",
    "CULTURE_REPO_ID",
    "SAE_REPO_ID",
    "GPT_SMALL",
    "QuizMachineConfig",
    "QM_CONFIG",
    "set_seed",
    "get_device",
    "num_params",
    "LOG_FILE",
    "log_string",
]

## Culture repo ##

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[1]
LOG_DIR = ROOT_DIR / "results_interp"

culture_dir = ROOT_DIR / "culture"
assert culture_dir.exists()
if str(culture_dir) not in sys.path:
    sys.path.append(str(culture_dir.resolve()))

CULTURE_REPO_ID = "tommyp111/culture-gpt"
SAE_REPO_ID = "tommyp111/culture-sae"
TOKENIZER_REPO_ID = "tommyp111/culture-tokenizer"

## Config ##

from transformer_lens import HookedTransformerConfig

GPT_SMALL = HookedTransformerConfig(
    d_model=512,
    d_head=64,
    d_mlp=2048,
    n_heads=8,
    n_layers=12,
    n_ctx=404,
    d_vocab=15,
    act_fn="relu",
    final_ln=False,  # requires custom patch
    tokenizer_name=TOKENIZER_REPO_ID,
    trust_remote_code=True,
    default_prepend_bos=False,
    tokenizer_prepends_bos=False,
)


from dataclasses import dataclass


@dataclass
class QuizMachineConfig:
    batch_size: int = 25
    inference_batch_size: int = 50
    nb_test_samples: int = 2000
    prompt_noise: float = 0.05


QM_CONFIG = QuizMachineConfig()

## Utils ##

import torch as t


def set_seed(seed=307451):
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


def get_device():
    if t.cuda.is_available():
        return t.device("cuda:0")
    elif t.backends.mps.is_available():
        return t.device("mps")
    else:
        return t.device("cpu")


def num_params(model):
    return sum(p.numel() for p in model.parameters())


import time

LOG_FILE = open(LOG_DIR / "eval.log", "a")


def log_string(s):
    t = time.strftime("%Y%m%d-%H:%M:%S ", time.localtime())

    if LOG_FILE is not None:
        LOG_FILE.write(t + s + "\n")
        LOG_FILE.flush()

    print(t + s)
    sys.stdout.flush()
