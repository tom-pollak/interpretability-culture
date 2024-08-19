# Interpreting Culture

> This project was partly made as an application for [MATS](https://www.matsprogram.org/) program

> This project builds upon [Francois Fleuret culture](https://fleuret.org/cgi-bin/gitweb/gitweb.cgi?p=culture.git;a=tree) (with his permission).

## Setup

```bash
# install my interp library & culture
pip install -r requirements.txt
pip install -e .

# patch TransformerLens
# culture MyGPT does not include a final layer norm, need to add support for this
cd TransformerLens
git apply ../transfomer_lens_final_ln.patch
pip install -e .
```

## Interpretability Report:

- [01_induction_heads.pynb](./01_induction_heads.ipynb)
- [02_sae.pynb](./02_sae.ipynb)


## Motivation / why

I've been following this project from [Francois Fleuret](https://x.com/francoisfleuret) for a while now.

https://fleuret.org/public/culture/draft-paper.pdf


The hypothesis is that intelligence emerges from "social competition" of different agents. The experiment trains 5 GPTs on programmatically generated 2D "world" quizzes, then once the models have sufficiently learned the task (accuracy > 95%) they attempt to generate their own "culture" quizzes. These quizzes are kept if 4/5 of the models agree on the answer, and one gets it wrong (to make it correct, but sufficiently difficult).

The idea is that the models will start producing progressively more difficult quizzes as a result, and (ideally) new and unique concepts through social interaction.

## How this relates to interpretability

I thought 5 GPTs trained in this group setting would be an interesting project from an interpretability perspective:

- Are there universal features shared across these models? like in [Universal Neurons in GPT2 Language Model](https://arxiv.org/abs/2401.12181)
- We can compare and contrast features learned using an SAE
    - This is similar version to the "train with different seeds" and compare features
- Since the quizzes are synthetic, it should be easier to "interpret" the behaviour of these models, since I can easily partition the data into the different tasks, and dynamically generate new unseen tasks.
- Fundamentally these models are the same as a normal GPT, so I can use the same interpretability tools to understand the models.
- The feature visualizations on these grid tasks (rather than text sentences) would look pretty cool


## Problem outline

Quite ambitious since the models were not compatible with TransformerLens library, and everything was implemented in a unique way. So there was a bit of a standard software engineering challenge to integrate it with existing interpretability tools.

I always try to do something like this with my own projects anyway -- implementing from scratch helps me learn much better than just `AutoModel.from_pretrained`.

## Reproducing Results

There's a few "gotachs" in running the GPTs:

- Added sinusoidal support for TransformerLens
- `use_past_kv_cache` is buggy, I _think_ this is from hacking in sinusoidal positional encoding?
- There's no final layer norm in `MyGPT`, so I had to patch TransformerLens to support this too
- You must prepend the input with a `0` as a BOS token (the models generate the entire sequence when creating new quizzes, but not for eval)

## Repository

### `culture.py`

Contains most of the lib functionality, including:

- `generate` function (using `model.generate`)
- `load_culture`: Loading the `MyGPT` models
- `load_hooked`: Converting `MyGPT` weights into a `HookedTransformer`
- `load_quizzes`: Loading the `QuizMachine` (culture quizzes)
- `run_tests`: Running tests on the models


Run `python -m interp.culture 0 1 2 3 --num_test_samples 100` to test most of the library functionality, and run accuracy tests on the models. Setting `--num_test_samples` to 2000 is standard for eval, and should achieve ~95% accuracy. I've found that using a smaller number of samples can give a lower accuracy (should be at least 80s).


### `grid_tokenizer.py`

Create a HF tokenizer for the models. required for `train_sae.py`

Can find it on HF here: [tommyp111/culture-grid-tokenizer](https://huggingface.co/tommyp111/culture-grid-tokenizer)

- `repr_grid` for pretty printing.
- `sinusoidal_positional_encoding` impl
- `TOK_PREPROCESS` & `prep_quiz` for preprocessing quizzes

### `dataset.py`

Create a 1M element HF dataset. Again used for `train_sae.py`

- [tommyp111/culture-quizzes-1M](https://huggingface.co/datasets/tommyp111/culture-puzzles-1M)
- [tommyp111/culture-quizzes-1M-partitioned](https://huggingface.co/datasets/tommyp111/culture-puzzles-1M-partitioned)


### `train_sae.py`

Train a sparse autoencoder on the models using `sae_lens.SAETrainingRunner`.
