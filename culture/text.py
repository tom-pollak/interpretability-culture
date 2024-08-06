#!/usr/bin/env python

import math

import torch, torchvision

from torch import nn
from torch.nn import functional as F

class Text:
    # returns a nb x (L+1+L) long tensor where L is the length of one
    # of the two states of a quizz
    def generate_seq(self, nb):
        all_tokens="<>123456789ABCDEF+-="
        self.tok2int=dict([(t, i) for i, t in enumerate(all_tokens)])
        self.int2tok=dict([(i, t) for i, t in enumerate(all_tokens)])
        result = []
        for n in range(nb):
            
        return self.txt2seq(["2A3B>AABBB"])

    # save a file to vizualize quizzes, you can save a txt or png file
    def save_quizzes(self, input, result_dir, filename_prefix):
        pass

    # returns a pair (forward_tokens, backward_token)
    def direction_tokens(self):
        return (0, 1)

    def seq2txt(self, seq):
        result = []
        for s in seq:
            result.append("".join([ self.int2tok[x.item()] for x in s]))
        return result

    def txt2seq(self, seq):
        result = []
        for s in seq:
            result.append(torch.tensor([ self.tok2int[x] for x in s])[None,:])
        return torch.cat(result, dim=0)

######################################################################

if __name__ == "__main__":
    problem = Text()
    seq = problem.generate_seq(10)
    for s in problem.seq2txt(seq):
        print(s)

