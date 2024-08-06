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
            [128, 0, 0],
            [0, 0, 128],
            [255, 0, 0],
            [0, 255, 0],
            [64, 64, 255],
            [255, 192, 0],
            [0, 255, 255],
            [255, 0, 255],
            [192, 255, 192],
            [255, 192, 192],
            [192, 192, 255],
            [192, 192, 192],
        ]
    )

    token_background = 0
    token_h_wall = 1
    token_v_wall = 2
    first_object_token = 3
    nb_object_tokens = colors.size(0) - first_object_token
    token_forward = first_object_token + nb_object_tokens
    token_backward = token_forward + 1

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
        frame_sequences = []

        for _ in tqdm.tqdm(range(nb), dynamic_ncols=True, desc="world generation"):
            result = torch.zeros(
                self.nb_iterations, self.height, self.width, dtype=torch.int64
            )

            for w in range(self.nb_walls):
                if torch.rand(1) < 0.5:
                    i = torch.randint(self.height, (1,))
                    while True:
                        j1, j2 = torch.randint(self.width, (2,))
                        if j2 >= j1 + 2 and j2 <= j1 + self.width // 2:
                            break
                    result[:, i, j1:j2] = self.token_v_wall
                else:
                    j = torch.randint(self.height, (1,))
                    while True:
                        i1, i2 = torch.randint(self.height, (2,))
                        if i2 >= i1 + 2 and i2 <= i1 + self.height // 2:
                            break
                    result[:, i1:i2, j] = self.token_v_wall

            i, j, vi, vj = (
                torch.empty(self.nb_objects, dtype=torch.int64),
                torch.empty(self.nb_objects, dtype=torch.int64),
                torch.empty(self.nb_objects, dtype=torch.int64),
                torch.empty(self.nb_objects, dtype=torch.int64),
            )

            col = (
                torch.randperm(self.nb_object_tokens)[: self.nb_objects].sort().values
                + self.first_object_token
            )

            for n in range(self.nb_objects):
                while True:
                    i[n] = torch.randint(self.height, (1,))
                    j[n] = torch.randint(self.width, (1,))
                    if result[0, i[n], j[n]] == 0:
                        break
                vm = torch.randint(4, (1,))
                vi[n], vj[n] = (vm % 2) * 2 - 1, (vm // 2) * 2 - 1

            for l in range(self.nb_iterations):
                for n in range(self.nb_objects):
                    c = col[n]
                    result[l, i[n], j[n]] = c
                    ni, nj = i[n] + vi[n], j[n] + vj[n]

                    if vi[n] == 0 or vj[n] == 0:
                        if (
                            result[0, ni, nj] == self.token_v_wall
                            or ni < 0
                            or nj < 0
                            or ni >= self.height
                            or nj >= self.width
                        ):
                            vi[n] = -vi[n]
                            ni = i[n] + vi[n]
                            vj[n] = -vj[n]
                            nj = j[n] + vj[n]

                    else:
                        #         jp  jj  jm
                        #        +---+---+---+
                        #    ip  |B  |A  |H  |
                        #        |   |   |   |
                        #        +---+---+---+
                        #    ii  |C  |+  |G  |
                        #        |   | \ |   |
                        #        +---+---+---+
                        #    im  |D  |E  |F  |
                        #        |   |   |   |
                        #        +---+---+---+

                        ip = i[n] + vi[n]
                        ii = i[n].clone()
                        im = i[n] - vi[n]
                        jp = j[n] + vj[n]
                        jj = j[n].clone()
                        jm = j[n] - vj[n]

                        out_ip = ip < 0 or ip >= self.height
                        out_ii = ii < 0 or ii >= self.height
                        out_im = im < 0 or im >= self.height
                        out_jp = jp < 0 or jp >= self.width
                        out_jj = jj < 0 or jj >= self.width
                        out_jm = jm < 0 or jm >= self.width

                        a = out_ip or out_jj or result[0, ip, jj] == self.token_v_wall
                        b = out_ip or out_jp or result[0, ip, jp] == self.token_v_wall
                        c = out_ii or out_jp or result[0, ii, jp] == self.token_v_wall
                        d = out_im or out_jp or result[0, im, jp] == self.token_v_wall
                        e = out_im or out_jj or result[0, im, jj] == self.token_v_wall
                        f = out_im or out_jm or result[0, im, jm] == self.token_v_wall
                        g = out_ii or out_jm or result[0, ii, jm] == self.token_v_wall
                        h = out_ip or out_jm or result[0, ip, jm] == self.token_v_wall

                        if (a and c) or (not a and not c and b):
                            vi[n] = -vi[n]
                            ni = i[n]  # + vi[n]
                            vj[n] = -vj[n]
                            nj = j[n]  # + vj[n]

                        if a and not c:
                            vi[n] = -vi[n]
                            ni = i[n]  # + vi[n]

                        if not a and c:
                            vj[n] = -vj[n]
                            nj = j[n]  # + vj[n]

                    assert abs(i[n] - ni) <= 1 and abs(j[n] - nj) <= 1
                    i[n], j[n] = ni, nj
                    # print(f"#6 {i[n]=} {j[n]=} {ni=} {nj=}")

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
    frame_sequences = sky.generate_frame_sequences(nb=36)
    delay = time.perf_counter() - start_time
    print(f"{frame_sequences.size(0)/delay:02f} seq/s")

    # print(sky.seq2str(seq[:4]))

    for t in range(frame_sequences.size(1)):
        img = sky.seq2img(frame_sequences[:, t])
        torchvision.utils.save_image(
            img.float() / 255.0,
            f"/tmp/frame_{t:03d}.png",
            nrow=6,
            padding=6,
            pad_value=0,
        )

    # m = (torch.rand(seq.size()) < 0.05).long()
    # seq = (1 - m) * seq + m * 23

    img = sky.seq2img(frame_sequences[:60])

    torchvision.utils.save_image(
        img.float() / 255.0, "/tmp/world.png", nrow=6, padding=10, pad_value=0
    )
