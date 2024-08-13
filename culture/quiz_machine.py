#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import math, os, tqdm, warnings, sys

import torch, torchvision

from torch import nn
from torch.nn import functional as F

import mygpt
from mygpt import BracketedSequence

import threading

from torch import nn
TOK_PREPROCESS = nn.ConstantPad1d((1, -1), value=0)  # pads with a 0 start token

######################################################################

# ar_mask is a tensor with 0s and 1s, of same shape as input, with
# 1s where tokens should be generated. The others are kept
# unchanged.


def one_batch_masked_inplace_autoregression(
    model,
    input,
    ar_mask,
    seq_logproba,
    deterministic_synthesis=False,
    use_brack_seq=True,
):
    if input.size(0) == 0:
        return

    to_generate = (ar_mask.sum(0) > 0).nonzero()

    if to_generate.min() > 0:
        inp = BracketedSequence(input, 0, to_generate.min())
        if use_brack_seq:
            model(inp)  # Needed to initialize the model's cache
        else:
            model(TOK_PREPROCESS(inp.slice()))
    for s in range(to_generate.min(), to_generate.max() + 1):
        inp = BracketedSequence(input, s, 1)
        if use_brack_seq:
            output = model(inp).x
        else:
            output = model(TOK_PREPROCESS(inp.x))

        logits = output[:, s]

        if deterministic_synthesis:
            t_next = logits.argmax(-1)
        else:
            dist = torch.distributions.categorical.Categorical(logits=logits)
            t_next = dist.sample()

        all_n = torch.arange(t_next.size(0))

        seq_logproba += logits[all_n, t_next]

        input[:, s] = ar_mask[:, s] * t_next + (1 - ar_mask[:, s]) * input[:, s]


######################################################################


class QuizMachine:
    def __init__(
        self,
        problem,
        batch_size,
        result_dir,
        prompt_noise,
        logger,
        use_brack_seq=True,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.problem = problem
        self.batch_size = batch_size
        self.device = device
        self.logger = logger
        self.use_brack_seq = use_brack_seq
        self.prompt_len = None
        self.answer_len = None
        self.prompt_noise = prompt_noise

        self.understood_structures = [
            (("A", "f_A", "B", "f_B"), (0, 0, 0, 1), (0, 0, 1, 0)),
            (("f_A", "A", "f_B", "B"), (0, 0, 0, 1), (0, 0, 1, 0)),
            (("B", "f_B", "A", "f_A"), (0, 0, 0, 1), (0, 0, 0, 0)),
            (("f_B", "B", "f_A", "A"), (0, 0, 0, 1), (0, 0, 0, 0)),
            (("f_B", "f_A", "A", "B"), (0, 1, 1, 1), (0, 0, 0, 0)),
        ]

        self.LOCK_C_QUIZZES = threading.Lock()
        self.train_c_quizzes = []
        self.test_c_quizzes = []

    def vocabulary_size(self):
        return self.problem.nb_token_values

    ######################################################################

    def autoregression(
        self,
        model,
        input,
        ar_mask,
        seq_logproba=None,
        progress_bar_desc=None,
    ):
        assert input.size() == ar_mask.size()

        if seq_logproba is None:
            seq_logproba = torch.empty(input.size(0), device=self.device)

        batches = zip(
            input.split(self.batch_size),
            ar_mask.split(self.batch_size),
            seq_logproba.split(self.batch_size),
        )

        if progress_bar_desc is not None:
            batches = tqdm.tqdm(
                batches,
                dynamic_ncols=True,
                desc=progress_bar_desc,
                total=(input.size(0) + self.batch_size - 1) // self.batch_size,
            )

        with torch.autograd.no_grad():
            t = model.training
            model.eval()

            for input, ar_mask, seq_logproba in batches:
                one_batch_masked_inplace_autoregression(
                    model=model,
                    input=input,
                    ar_mask=ar_mask,
                    seq_logproba=seq_logproba,
                    deterministic_synthesis=False,
                    use_brack_seq=self.use_brack_seq
                )

            model.train(t)

    ######################################################################

    def data_input(self, model, split="train"):
        assert split in {"train", "test"}

        with self.LOCK_C_QUIZZES:
            if split == "train":
                w_quizzes = model.train_w_quizzes
                c_quizzes = self.train_c_quizzes
            else:
                w_quizzes = model.test_w_quizzes
                c_quizzes = self.test_c_quizzes

            if len(c_quizzes) > 0:
                c_quizzes = torch.cat(c_quizzes, dim=0)

                if c_quizzes.size(0) > w_quizzes.size(0) // 2:
                    i = torch.randperm(c_quizzes.size(0))[: w_quizzes.size(0) // 2]
                    c_quizzes = c_quizzes[i]

                i = torch.randperm(w_quizzes.size(0))[
                    : w_quizzes.size(0) - c_quizzes.size(0)
                ]
                w_quizzes = w_quizzes[i]

                quizzes = torch.cat([w_quizzes, c_quizzes], dim=0)
                from_w = torch.arange(
                    quizzes.size(0), device=quizzes.device
                ) < w_quizzes.size(0)

            else:
                quizzes = w_quizzes.clone()
                from_w = torch.full((quizzes.size(0),), True, device=quizzes.device)

        i = torch.randperm(quizzes.size(0), device=quizzes.device)
        quizzes, from_w = quizzes[i], from_w[i]

        self.randomize_configuations_inplace(
            quizzes, structs=[s for s, m, _ in self.understood_structures]
        )

        if self.prompt_noise > 0.0:
            for struct, mask, noise_mask in self.understood_structures:
                i = self.problem.indices_select(quizzes=quizzes, struct=struct)
                if i.any():
                    quizzes[i] = self.problem.inject_noise(
                        quizzes[i], self.prompt_noise, struct=struct, mask=noise_mask
                    )

        return quizzes, from_w

    ######################################################################

    def make_ar_mask(self, quizzes, struct, mask):
        assert struct in [s for s, _, _ in self.understood_structures]
        return self.problem.make_ar_mask(quizzes, struct=struct, mask=mask)

    ######################################################################

    def predict(self, model, quizzes, struct, mask):
        ar_mask = self.make_ar_mask(quizzes=quizzes, struct=struct, mask=mask)
        result = quizzes * (1 - ar_mask)

        seq_logproba = torch.empty(quizzes.size(0), device=self.device)

        self.autoregression(
            model=model,
            input=result,
            ar_mask=ar_mask,
            seq_logproba=seq_logproba,
            progress_bar_desc="accuracy",
        )

        correct = (result == quizzes).min(dim=1).values.long()

        return result, correct

    ######################################################################

    def produce_results(self, n_epoch, model, input, result_dir):
        input = input.to(self.device)
        result = input.new(input.size())
        correct = input.new(input.size(0))
        predicted_parts = input.new(input.size(0), 4)

        nb = 0

        # We consider all the configurations that we train for
        for struct, mask, _ in self.understood_structures:
            i = self.problem.indices_select(quizzes=input, struct=struct)
            nb += i.long().sum()
            result[i], correct[i] = self.predict(
                model=model, quizzes=input[i], struct=struct, mask=mask
            )
            predicted_parts[i] = torch.tensor(mask, device=self.device)[None, :]
            solution_is_deterministic = predicted_parts[i].sum(dim=-1) == 1
            correct[i] = (2 * correct[i] - 1) * (solution_is_deterministic).long()

        assert nb == input.size(0)

        nb_correct = (correct == 1).long().sum()
        nb_total = (correct != 0).long().sum()
        self.logger(
            f"test_accuracy {n_epoch} model {model.id} val {nb_correct} / {nb_total}"
        )

        main_test_accuracy = nb_correct / nb_total

        ##############################

        correct_parts = predicted_parts * correct[:, None]

        result = result[:128]
        predicted_parts = predicted_parts[:128]
        correct_parts = correct_parts[:128]

        self.problem.save_quizzes_as_image(
            result_dir,
            f"culture_prediction_{n_epoch:04d}_{model.id:02d}.png",
            quizzes=result,
            predicted_parts=predicted_parts,
            correct_parts=correct_parts,
        )

        return main_test_accuracy

    ######################################################################

    def randomize_configuations_inplace(self, quizzes, structs):
        r = torch.randint(len(structs), (quizzes.size(0),), device=quizzes.device)
        for c in range(len(structs)):
            quizzes[r == c] = self.problem.reconfigure(
                quizzes[r == c], struct=structs[c]
            )

    ######################################################################

    def renew_train_w_quizzes(self, model):
        if hasattr(model, "hard_w_quizzes"):
            hard_w_quizzes = self.problem.reconfigure(
                model.hard_w_quizzes, struct=("A", "f_A", "B", "f_B")
            )
            self.logger(
                f"re-using {hard_w_quizzes.size(0)} hard world quizzes from model {model.id}"
            )
            if hard_w_quizzes.size(0) >= model.train_w_quizzes.size(0):
                nb_to_generate = 0
                model.train_w_quizzes[...] = hard_w_quizzes[
                    torch.randperm(hard_w_quizzes.size(0))[
                        model.train_w_quizzes.size(0)
                    ]
                ]
            else:
                nb_to_generate = model.train_w_quizzes.size(0) - hard_w_quizzes.size(0)
                model.train_w_quizzes[...] = torch.cat(
                    [
                        hard_w_quizzes,
                        self.problem.generate_w_quizzes(nb_to_generate),
                    ],
                    dim=0,
                )
        else:
            nb_to_generate = 0
            model.train_w_quizzes[...] = self.problem.generate_w_quizzes(
                model.train_w_quizzes.size(0)
            )

    ######################################################################

    def store_c_quizzes(self, new_c_quizzes, for_train=True):
        with self.LOCK_C_QUIZZES:
            if for_train:
                self.train_c_quizzes.append(new_c_quizzes.to("cpu"))
            else:
                self.test_c_quizzes.append(new_c_quizzes.to("cpu"))

    def save_c_quizzes(self, filename):
        torch.save((self.train_c_quizzes, self.test_c_quizzes), filename)

    def load_c_quizzes(self, filename):
        self.train_c_quizzes, self.test_c_quizzes = torch.load(filename, weights_only=False)

    ######################################################################

    def models_logprobas(
        self,
        models_for_validation,
        c_quizzes,
        struct,
        mask,
        noise_mask=None,
        device=None,
    ):
        if device is None:
            device = self.device

        c_quizzes = self.problem.reconfigure(c_quizzes, struct)

        seq_logproba = torch.zeros(
            c_quizzes.size(0),
            max([m.id for m in models_for_validation]) + 1,
            device=device,
        )

        if self.prompt_noise > 0.0 and noise_mask is not None:
            c_quizzes = self.problem.inject_noise(
                c_quizzes, self.prompt_noise, struct=struct, mask=noise_mask
            )

        for model in models_for_validation:
            with torch.autograd.no_grad():
                t = model.training
                model.eval()

                for input, l in zip(
                    c_quizzes.split(self.batch_size),
                    seq_logproba.split(self.batch_size),
                ):
                    input = input.to(device)
                    ar_mask = self.make_ar_mask(input, struct=struct, mask=mask)
                    if self.use_brack_seq:
                        output = model(mygpt.BracketedSequence(input)).x
                    else:
                        output = model(input)
                    l[:, model.id] = (
                        -F.cross_entropy(
                            output.transpose(1, 2), input, reduction="none"
                        )
                        * ar_mask
                    ).sum(dim=1)

                model.train(t)

        return seq_logproba.to("cpu")

    ######################################################################

    def generate_c_quizzes(
        self, nb, model_for_generation, procedure, to_recycle=None, recorder=None
    ):
        seq_logproba = torch.zeros(nb, device=self.device)

        c_quizzes = None

        for s, m, mt in procedure:
            if c_quizzes is None:
                c_quizzes = self.problem.create_empty_quizzes(nb, s)
                c_quizzes = c_quizzes.to(self.device)
            elif s != pred_s:
                c_quizzes = self.problem.reconfigure(c_quizzes, s)
            pred_s = s

            if mt is not None:
                mt(model_for_generation)

            self.autoregression(
                model=model_for_generation,
                input=c_quizzes,
                ar_mask=self.make_ar_mask(c_quizzes, s, m),
                seq_logproba=seq_logproba,
            )

            model_for_generation.reset_transformations()

            if recorder is not None:
                x = c_quizzes.clone()
                t = torch.tensor(m, device=x.device)[None, :].expand(x.size(0), -1)
                recorder.append(
                    self.problem.reconfigure([x, t], ("A", "f_A", "B", "f_B"))
                )

            if to_recycle is not None and to_recycle.size(0) > 0:
                to_recycle = self.problem.reconfigure(to_recycle, s)
                c_quizzes[: to_recycle.size(0)] = to_recycle

            to_recycle = None

        c_quizzes = self.problem.reconfigure(c_quizzes, ("A", "f_A", "B", "f_B"))

        return c_quizzes.to("cpu")

    ######################################################################
