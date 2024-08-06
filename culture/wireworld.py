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


class Wireworld(problem.Problem):
    colors = torch.tensor(
        [
            [128, 128, 128],
            [128, 128, 255],
            [255, 0, 0],
            [255, 255, 0],
        ]
    )

    token_empty = 0
    token_head = 1
    token_tail = 2
    token_conductor = 3
    token_forward = 4
    token_backward = 5

    token2char = (
        "_" + "".join([chr(ord("A") + n) for n in range(len(colors) - 1)]) + "><"
    )

    def __init__(
        self, height=6, width=8, nb_objects=2, nb_walls=2, speed=1, nb_iterations=4
    ):
        self.height = height
        self.width = width
        self.nb_objects = nb_objects
        self.nb_walls = nb_walls
        self.speed = speed
        self.nb_iterations = nb_iterations

    def direction_tokens(self):
        return self.token_forward, self.token_backward

    def generate_frame_sequences(self, nb):
        result = []
        N = 100
        for _ in tqdm.tqdm(
            range(0, nb + N, N), dynamic_ncols=True, desc="world generation"
        ):
            result.append(self.generate_frame_sequences_hard(100))
        return torch.cat(result, dim=0)[:nb]

    def generate_frame_sequences_hard(self, nb):
        frame_sequences = []
        nb_frames = (self.nb_iterations - 1) * self.speed + 1

        result = torch.full(
            (nb * 4, nb_frames, self.height, self.width),
            self.token_empty,
        )

        for n in range(result.size(0)):
            while True:
                i = torch.randint(self.height, (1,))
                j = torch.randint(self.width, (1,))
                v = torch.randint(2, (2,))
                vi = v[0] * (v[1] * 2 - 1)
                vj = (1 - v[0]) * (v[1] * 2 - 1)
                while True:
                    if i < 0 or i >= self.height or j < 0 or j >= self.width:
                        break
                    o = 0
                    if i > 0:
                        o += (result[n, 0, i - 1, j] == self.token_conductor).long()
                    if i < self.height - 1:
                        o += (result[n, 0, i + 1, j] == self.token_conductor).long()
                    if j > 0:
                        o += (result[n, 0, i, j - 1] == self.token_conductor).long()
                    if j < self.width - 1:
                        o += (result[n, 0, i, j + 1] == self.token_conductor).long()
                    if o > 1:
                        break
                    result[n, 0, i, j] = self.token_conductor
                    i += vi
                    j += vj
                if (
                    result[n, 0] == self.token_conductor
                ).long().sum() > self.width and torch.rand(1) < 0.5:
                    break

            while True:
                for _ in range(self.height * self.width):
                    i = torch.randint(self.height, (1,))
                    j = torch.randint(self.width, (1,))
                    v = torch.randint(2, (2,))
                    vi = v[0] * (v[1] * 2 - 1)
                    vj = (1 - v[0]) * (v[1] * 2 - 1)
                    if (
                        i + vi >= 0
                        and i + vi < self.height
                        and j + vj >= 0
                        and j + vj < self.width
                        and result[n, 0, i, j] == self.token_conductor
                        and result[n, 0, i + vi, j + vj] == self.token_conductor
                    ):
                        result[n, 0, i, j] = self.token_head
                        result[n, 0, i + vi, j + vj] = self.token_tail
                        break

                # if torch.rand(1) < 0.75:
                break

        weight = torch.full((1, 1, 3, 3), 1.0)

        mask = (torch.rand(result[:, 0].size()) < 0.01).long()
        rand = torch.randint(4, mask.size())
        result[:, 0] = mask * rand + (1 - mask) * result[:, 0]

        # empty->empty
        # head->tail
        # tail->conductor
        # conductor->head if 1 or 2 head in the neighborhood, or remains conductor

        nb_heads = (result[:, 0] == self.token_head).flatten(1).long().sum(dim=1)
        valid = nb_heads > 0

        for l in range(nb_frames - 1):
            nb_head_neighbors = (
                F.conv2d(
                    input=(result[:, l] == self.token_head).float()[:, None, :, :],
                    weight=weight,
                    padding=1,
                )
                .long()
                .squeeze(1)
            )
            mask_1_or_2_heads = (nb_head_neighbors == 1).long() + (
                nb_head_neighbors == 2
            ).long()
            result[:, l + 1] = (
                (result[:, l] == self.token_empty).long() * self.token_empty
                + (result[:, l] == self.token_head).long() * self.token_tail
                + (result[:, l] == self.token_tail).long() * self.token_conductor
                + (result[:, l] == self.token_conductor).long()
                * (
                    mask_1_or_2_heads * self.token_head
                    + (1 - mask_1_or_2_heads) * self.token_conductor
                )
            )
            pred_nb_heads = nb_heads
            nb_heads = (
                (result[:, l + 1] == self.token_head).flatten(1).long().sum(dim=1)
            )
            valid = torch.logical_and(valid, (nb_heads >= pred_nb_heads))

        result = result[valid]

        result = result[
            :, torch.arange(self.nb_iterations, device=result.device) * self.speed
        ]

        i = (result[:, -1] == self.token_head).flatten(1).max(dim=1).values > 0
        result = result[i]

        # print(f"{result.size(0)=} {nb=}")

        if result.size(0) < nb:
            # print(result.size(0))
            result = torch.cat(
                [result, self.generate_frame_sequences(nb - result.size(0))], dim=0
            )

        return result[:nb]

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
        m = torch.logical_and(x >= 0, x < 4).long()

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

    wireworld = Wireworld(height=8, width=10, nb_iterations=5, speed=1)

    start_time = time.perf_counter()
    frame_sequences = wireworld.generate_frame_sequences(nb=96)
    delay = time.perf_counter() - start_time
    print(f"{frame_sequences.size(0)/delay:02f} seq/s")

    # print(wireworld.seq2str(seq[:4]))

    for t in range(frame_sequences.size(1)):
        img = wireworld.seq2img(frame_sequences[:, t])
        torchvision.utils.save_image(
            img.float() / 255.0,
            f"/tmp/frame_{t:03d}.png",
            nrow=8,
            padding=6,
            pad_value=0,
        )

    # m = (torch.rand(seq.size()) < 0.05).long()
    # seq = (1 - m) * seq + m * 23

    wireworld = Wireworld(height=8, width=10, nb_iterations=2, speed=5)
    token_sequences = wireworld.generate_token_sequences(32)
    wireworld.save_quizzes(token_sequences, "/tmp", "seq")
    # img = wireworld.seq2img(frame_sequences[:60])

    # torchvision.utils.save_image(
    # img.float() / 255.0, "/tmp/world.png", nrow=6, padding=10, pad_value=0.1
    # )
