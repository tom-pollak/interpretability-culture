#!/usr/bin/env python

import math

import torch, torchvision

from torch import nn
from torch.nn import functional as F

import problem

class MathWorld:
    def __init__(self):

    def nb_token_values(self):
        pass

    def trivial_prompts_and_answers(self, prompts, answers):
        pass

    def vars(self, n):
        return "ABCDEF"[torch.randperm(6)]


    def task_commute(self):
        vars

    def generate_prompts_and_answers_(self, nb):
        result = []

        for _ in range(nb):
            result.append(self.task_commute())

        pass

    # save a file to vizualize quizzes, you can save a txt or png file
    def save_quiz_illustrations(
        self,
        result_dir,
        filename_prefix,
        prompts,
        answers,
        predicted_prompts=None,
        predicted_answers=None,
    ):
        pass

    def save_some_examples(self, result_dir):
        pass
