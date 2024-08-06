#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import math, sys, tqdm, os

import torch, torchvision

from torch import nn
from torch.nn import functional as F

######################################################################

import problem


class Physics(problem.Problem):
    colors = torch.tensor(
        [
            [255, 255, 255],
            [0, 0, 0],
        ]
    )

    token_zero = 0
    token_one = 1
    token_forward = 2
    token_backward = 3

    token2char = "_X><"

    def __init__(self, height=9, width=9, init_proba=0.1, nb_iterations=4):
        self.height = height
        self.width = width
        self.init_proba = init_proba
        self.nb_iterations = nb_iterations

    def direction_tokens(self):
        return self.token_forward, self.token_backward

    def starting_configuration(self, nb):
        conf = (torch.rand(nb, self.height, self.width) < proba_init).long()
        return conf * token_one + (1 - conf) * self.token_zero

    def critters_step(self, state, di, dj):
        

    def generate_frame_sequences(self, nb):
        result = torch.zeros(
            nb, self.nb_iterations, self.height+2, self.width+2, dtype=torch.int64
        )

        result[:, 0] = starting_configuration(nb)

        frame_sequences.append(result[None])

        return torch.cat(frame_sequences, dim=0)

    def generate_token_sequences(self, nb):
        frame_sequences = self.generate_frame_sequences(nb)

        result = []

        for frame_sequence in frame_sequences:
            a = []
            if torch.rand(1) < 0.5:
                for frame in frame_sequence:
                    if len(a) > 0:
                        a.append(torch.tensor([self.token_forward]))
                    a.append(frame.flatten())
            else:
                for frame in reversed(frame_sequence):
                    if len(a) > 0:
                        a.append(torch.tensor([self.token_backward]))
                    a.append(frame.flatten())

            result.append(torch.cat(a, dim=0)[None, :])

        return torch.cat(result, dim=0)

    ######################################################################

    def frame2img(self, x, scale=15):
        x = x.reshape(-1, self.height, self.width)
        m = torch.logical_and(
            x >= 0, x < self.first_object_token + self.nb_object_tokens
        ).long()

        x = self.colors[x * m].permute(0, 3, 1, 2)
        s = x.shape
        x = x[:, :, :, None, :, None].expand(-1, -1, -1, scale, -1, scale)
        x = x.reshape(s[0], s[1], s[2] * scale, s[3] * scale)

        x[:, :, :, torch.arange(0, x.size(3), scale)] = 0
        x[:, :, torch.arange(0, x.size(2), scale), :] = 0
        x = x[:, :, 1:, 1:]

        for n in range(m.size(0)):
            for i in range(m.size(1)):
                for j in range(m.size(2)):
                    if m[n, i, j] == 0:
                        for k in range(2, scale - 2):
                            for l in [0, 1]:
                                x[n, :, i * scale + k, j * scale + k - l] = 0
                                x[
                                    n, :, i * scale + scale - 1 - k, j * scale + k - l
                                ] = 0

        return x

    def seq2img(self, seq, scale=15):
        all = [
            self.frame2img(
                seq[:, : self.height * self.width].reshape(-1, self.height, self.width),
                scale,
            )
        ]

        separator = torch.full((seq.size(0), 3, self.height * scale - 1, 1), 0)

        t = self.height * self.width

        while t < seq.size(1):
            direction_tokens = seq[:, t]
            t += 1

            direction_images = self.colors[
                torch.full(
                    (direction_tokens.size(0), self.height * scale - 1, scale), 0
                )
            ].permute(0, 3, 1, 2)

            for n in range(direction_tokens.size(0)):
                if direction_tokens[n] == self.token_forward:
                    for k in range(scale):
                        for l in [0, 1]:
                            direction_images[
                                n,
                                :,
                                (self.height * scale) // 2 - scale // 2 + k - l,
                                3 + scale // 2 - abs(k - scale // 2),
                            ] = 0
                elif direction_tokens[n] == self.token_backward:
                    for k in range(scale):
                        for l in [0, 1]:
                            direction_images[
                                n,
                                :,
                                (self.height * scale) // 2 - scale // 2 + k - l,
                                3 + abs(k - scale // 2),
                            ] = 0
                else:
                    for k in range(2, scale - 2):
                        for l in [0, 1]:
                            direction_images[
                                n,
                                :,
                                (self.height * scale) // 2 - scale // 2 + k - l,
                                k,
                            ] = 0
                            direction_images[
                                n,
                                :,
                                (self.height * scale) // 2 - scale // 2 + k - l,
                                scale - 1 - k,
                            ] = 0

            all += [
                separator,
                direction_images,
                separator,
                self.frame2img(
                    seq[:, t : t + self.height * self.width].reshape(
                        -1, self.height, self.width
                    ),
                    scale,
                ),
            ]

            t += self.height * self.width

        return torch.cat(all, dim=3)

    def seq2str(self, seq):
        result = []
        for s in seq:
            result.append("".join([self.token2char[v] for v in s]))
        return result

    def save_image(self, input, result_dir, filename):
        img = self.seq2img(input.to("cpu"))
        image_name = os.path.join(result_dir, filename)
        torchvision.utils.save_image(img.float() / 255.0, image_name, nrow=6, padding=4)

    def save_quizzes(self, input, result_dir, filename_prefix):
        self.save_image(input, result_dir, filename_prefix + ".png")


######################################################################

if __name__ == "__main__":
    import time

    sky = Physics(height=10, width=15, speed=1, nb_iterations=100)

    start_time = time.perf_counter()
    frame_sequences = sky.generate_frame_sequences(nb=30)
    delay = time.perf_counter() - start_time
    print(f"{frame_sequences.size(0)/delay:02f} seq/s")

    # print(sky.seq2str(seq[:4]))

    for t in range(frame_sequences.size(1)):
        img = sky.seq2img(frame_sequences[:, t])
        torchvision.utils.save_image(
            img.float() / 255.0,
            f"/tmp/frame_{t:03d}.png",
            nrow=8,
            padding=6,
            pad_value=0,
        )

    # m = (torch.rand(seq.size()) < 0.05).long()
    # seq = (1 - m) * seq + m * 23

    img = sky.seq2img(frame_sequences[:10])

    torchvision.utils.save_image(
        img.float() / 255.0, "/tmp/world.png", nrow=8, padding=10, pad_value=0
    )
