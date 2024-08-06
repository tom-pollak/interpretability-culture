#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import math, sys, tqdm, os, warnings

import torch, torchvision

from torch import nn
from torch.nn import functional as F

######################################################################

import problem


def grow_islands(nb, height, width, nb_seeds, nb_iterations):
    w = torch.empty(5, 1, 3, 3)

    w[0, 0] = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    w[1, 0] = torch.tensor(
        [
            [-1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    w[2, 0] = torch.tensor(
        [
            [0.0, 1.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )

    w[3, 0] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, -1.0],
        ]
    )

    w[4, 0] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
        ]
    )

    Z = torch.zeros(nb, height, width)
    U = Z.flatten(1)

    for _ in range(nb_seeds):
        M = F.conv2d(Z[:, None, :, :], w, padding=1)
        M = torch.cat([M[:, :1], M[:, 1:].min(dim=1, keepdim=True).values], dim=1)
        M = ((M[:, 0] == 0) & (Z == 0)).long()
        Q = (M.flatten(1).max(dim=1).values > 0).long()[:, None]
        M = M * torch.rand(M.size())
        M = M.flatten(1)
        M = F.one_hot(M.argmax(dim=1), num_classes=M.size(1))
        U += M * Q

    for _ in range(nb_iterations):
        M = F.conv2d(Z[:, None, :, :], w, padding=1)
        M = torch.cat([M[:, :1], M[:, 1:].min(dim=1, keepdim=True).values], dim=1)
        M = ((M[:, 1] >= 0) & (Z == 0)).long()
        Q = (M.flatten(1).max(dim=1).values > 0).long()[:, None]
        M = M * torch.rand(M.size())
        M = M.flatten(1)
        M = F.one_hot(M.argmax(dim=1), num_classes=M.size(1))
        U = Z.flatten(1)
        U += M * Q

    M = Z.clone()
    Z = Z * (torch.arange(Z.size(1) * Z.size(2)) + 1).reshape(1, Z.size(1), Z.size(2))

    while True:
        W = Z.clone()
        Z = F.max_pool2d(Z, 3, 1, 1) * M
        if Z.equal(W):
            break

    Z = Z.long()
    U = Z.flatten(1)
    V = F.one_hot(U).max(dim=1).values
    W = V.cumsum(dim=1) - V
    N = torch.arange(Z.size(0))[:, None, None].expand_as(Z)
    Z = W[N, Z]

    return Z


class Grids(problem.Problem):
    named_colors = [
        ("white", [255, 255, 255]),
        ("red", [255, 0, 0]),
        ("green", [0, 192, 0]),
        ("blue", [0, 0, 255]),
        ("yellow", [255, 224, 0]),
        ("cyan", [0, 255, 255]),
        ("violet", [224, 128, 255]),
        ("lightgreen", [192, 255, 192]),
        ("brown", [165, 42, 42]),
        ("lightblue", [192, 192, 255]),
        ("gray", [128, 128, 128]),
    ]

    def __init__(
        self,
        max_nb_cached_chunks=None,
        chunk_size=None,
        nb_threads=-1,
        tasks=None,
    ):
        self.colors = torch.tensor([c for _, c in self.named_colors])
        self.height = 10
        self.width = 10
        self.cache_rec_coo = {}

        all_tasks = [
            self.task_replace_color,
            self.task_translate,
            self.task_grow,
            self.task_half_fill,
            self.task_frame,
            self.task_detect,
            self.task_count,
            self.task_trajectory,
            self.task_bounce,
            self.task_scale,
            self.task_symbols,
            self.task_isometry,
            self.task_islands,
        ]

        if tasks is None:
            self.all_tasks = all_tasks
        else:
            self.all_tasks = [getattr(self, "task_" + t) for t in tasks.split(",")]

        super().__init__(max_nb_cached_chunks, chunk_size, nb_threads)

    ######################################################################

    def frame2img(self, x, scale=15):
        x = x.reshape(x.size(0), self.height, -1)
        m = torch.logical_and(x >= 0, x < self.nb_token_values()).long()
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

    def save_image(
        self,
        result_dir,
        filename,
        prompts,
        answers,
        predicted_prompts=None,
        predicted_answers=None,
        nrow=4,
        margin=8,
    ):
        S = self.height * self.width
        As = prompts[:, 0 * (S + 1) : 0 * (S + 1) + S].view(-1, self.height, self.width)
        f_As = prompts[:, 1 * (S + 1) : 1 * (S + 1) + S].view(
            -1, self.height, self.width
        )
        Bs = prompts[:, 2 * (S + 1) : 2 * (S + 1) + S].view(-1, self.height, self.width)
        prompts = torch.cat([As, f_As, Bs], dim=2)
        answers = answers.reshape(answers.size(0), self.height, self.width)

        if predicted_prompts is None:
            predicted_prompts = 255

        if predicted_answers is None:
            predicted_answers = 255

        def add_frame(x, c, margin, bottom=False):
            if bottom:
                h, w, di, dj = x.size(2) + margin, x.size(3), 0, 0
            else:
                h, w, di, dj = (
                    x.size(2) + 2 * margin,
                    x.size(3) + 2 * margin,
                    margin,
                    margin,
                )

            y = x.new_full((x.size(0), x.size(1), h, w), 0)

            if type(c) is int:
                y[...] = c
            else:
                c = c.long()[:, None]
                c = (
                    (1 - ((c == 1).long() + (c == 0).long() + (c == -1).long()))
                    * torch.tensor([64, 64, 64])
                    + (c == 1).long() * torch.tensor([0, 255, 0])
                    + (c == 0).long() * torch.tensor([255, 255, 255])
                    + (c == -1).long() * torch.tensor([255, 0, 0])
                )
                y[...] = c[:, :, None, None]

            y[:, :, di : di + x.size(2), dj : dj + x.size(3)] = x

            return y

        img_prompts = torch.cat(
            [
                add_frame(
                    add_frame(self.frame2img(x), c=0, margin=1),
                    c=predicted_prompts,
                    margin=margin,
                )
                for x in prompts.to("cpu").split(split_size=self.width, dim=2)
            ],
            dim=3,
        )

        h = img_prompts.size(2)
        img_answers = add_frame(
            add_frame(self.frame2img(answers.to("cpu")), c=0, margin=1),
            c=predicted_answers,
            margin=margin,
        )

        separator_size = 2 * margin

        separator = img_prompts.new_full(
            (
                img_prompts.size(0),
                img_prompts.size(1),
                img_prompts.size(2),
                separator_size,
            ),
            255,
        )

        marker = img_prompts.new_full(
            (
                img_prompts.size(0),
                img_prompts.size(1),
                img_prompts.size(2),
                separator_size,
            ),
            255,
        )

        # marker[:, :, 0] = 0
        # marker[:, :, h - 1] = 0

        for k in range(1, 2 * separator_size - 8):
            i = k - (separator_size - 4)
            j = separator_size - 5 - abs(i)
            marker[:, :, h // 2 - 1 + i, 2 + j] = 0
            marker[:, :, h // 2 - 1 + i + 1, 2 + j] = 0

        img = torch.cat(
            [
                img_prompts,
                marker,
                img_answers,
            ],
            dim=3,
        )

        image_name = os.path.join(result_dir, filename)
        torchvision.utils.save_image(
            img.float() / 255.0,
            image_name,
            nrow=nrow,
            padding=margin * 4,
            pad_value=1.0,
        )

    ######################################################################

    def nb_token_values(self):
        return len(self.colors)

    # @torch.compile
    def rec_coo(
        self,
        nb_rec,
        min_height=3,
        min_width=3,
        surface_max=None,
        prevent_overlap=False,
    ):
        if surface_max is None:
            surface_max = self.height * self.width // 2

        signature = (nb_rec, min_height, min_width, surface_max)

        try:
            return self.cache_rec_coo[signature].pop()
        except IndexError:
            pass
        except KeyError:
            pass

        N = 10000
        while True:
            while True:
                i = torch.randint(self.height, (N * nb_rec, 2)).sort(dim=-1).values
                j = torch.randint(self.width, (N * nb_rec, 2)).sort(dim=-1).values

                big_enough = (
                    (i[:, 1] >= i[:, 0] + min_height)
                    & (j[:, 1] >= j[:, 0] + min_height)
                    & ((i[:, 1] - i[:, 0]) * (j[:, 1] - j[:, 0]) <= surface_max)
                )

                i, j = i[big_enough], j[big_enough]

                n = i.size(0) - i.size(0) % nb_rec

                if n > 0:
                    break

            i = i[:n].reshape(n // nb_rec, nb_rec, -1)
            j = j[:n].reshape(n // nb_rec, nb_rec, -1)

            if prevent_overlap:
                can_fit = ((i[:, :, 1] - i[:, :, 0]) * (j[:, :, 1] - j[:, :, 0])).sum(
                    dim=-1
                ) <= self.height * self.width
                i, j = i[can_fit], j[can_fit]
                if nb_rec == 2:
                    A_i1, A_i2, A_j1, A_j2 = (
                        i[:, 0, 0],
                        i[:, 0, 1],
                        j[:, 0, 0],
                        j[:, 0, 1],
                    )
                    B_i1, B_i2, B_j1, B_j2 = (
                        i[:, 1, 0],
                        i[:, 1, 1],
                        j[:, 1, 0],
                        j[:, 1, 1],
                    )
                    no_overlap = torch.logical_not(
                        (A_i1 >= B_i2)
                        & (A_i2 <= B_i1)
                        & (A_j1 >= B_j1)
                        & (A_j2 <= B_j1)
                    )
                    i, j = i[no_overlap], j[no_overlap]
                elif nb_rec == 3:
                    A_i1, A_i2, A_j1, A_j2 = (
                        i[:, 0, 0],
                        i[:, 0, 1],
                        j[:, 0, 0],
                        j[:, 0, 1],
                    )
                    B_i1, B_i2, B_j1, B_j2 = (
                        i[:, 1, 0],
                        i[:, 1, 1],
                        j[:, 1, 0],
                        j[:, 1, 1],
                    )
                    C_i1, C_i2, C_j1, C_j2 = (
                        i[:, 2, 0],
                        i[:, 2, 1],
                        j[:, 2, 0],
                        j[:, 2, 1],
                    )
                    no_overlap = (
                        (
                            (A_i1 >= B_i2)
                            | (A_i2 <= B_i1)
                            | (A_j1 >= B_j2)
                            | (A_j2 <= B_j1)
                        )
                        & (
                            (A_i1 >= C_i2)
                            | (A_i2 <= C_i1)
                            | (A_j1 >= C_j2)
                            | (A_j2 <= C_j1)
                        )
                        & (
                            (B_i1 >= C_i2)
                            | (B_i2 <= C_i1)
                            | (B_j1 >= C_j2)
                            | (B_j2 <= C_j1)
                        )
                    )
                    i, j = (i[no_overlap], j[no_overlap])
                else:
                    assert nb_rec == 1

            if i.size(0) > 1:
                break

        self.cache_rec_coo[signature] = [
            [
                (
                    i[n, k, 0].item(),
                    j[n, k, 0].item(),
                    i[n, k, 1].item(),
                    j[n, k, 1].item(),
                )
                for k in range(nb_rec)
            ]
            for n in range(i.size(0))
        ]

        return self.cache_rec_coo[signature].pop()

    ######################################################################

    # @torch.compile
    def task_replace_color(self, A, f_A, B, f_B):
        nb_rec = 3
        c = torch.randperm(len(self.colors) - 1)[: nb_rec + 1] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            r = self.rec_coo(nb_rec, prevent_overlap=True)
            for n in range(nb_rec):
                i1, j1, i2, j2 = r[n]
                X[i1:i2, j1:j2] = c[n]
                f_X[i1:i2, j1:j2] = c[n if n > 0 else -1]

    # @torch.compile
    def task_translate(self, A, f_A, B, f_B):
        while True:
            di, dj = torch.randint(3, (2,)) - 1
            if di.abs() + dj.abs() > 0:
                break

        nb_rec = 3
        c = torch.randperm(len(self.colors) - 1)[:nb_rec] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            while True:
                r = self.rec_coo(nb_rec, prevent_overlap=True)
                i1, j1, i2, j2 = r[nb_rec - 1]
                if (
                    i1 + di >= 0
                    and i2 + di < X.size(0)
                    and j1 + dj >= 0
                    and j2 + dj < X.size(1)
                ):
                    break

            for n in range(nb_rec):
                i1, j1, i2, j2 = r[n]
                X[i1:i2, j1:j2] = c[n]
                if n == nb_rec - 1:
                    f_X[i1 + di : i2 + di, j1 + dj : j2 + dj] = c[n]
                else:
                    f_X[i1:i2, j1:j2] = c[n]

    # @torch.compile
    def task_grow(self, A, f_A, B, f_B):
        di, dj = torch.randint(2, (2,)) * 2 - 1
        nb_rec = 3
        c = torch.randperm(len(self.colors) - 1)[:nb_rec] + 1
        direction = torch.randint(2, (1,)).item()
        for X, f_X in [(A, f_A), (B, f_B)]:
            while True:
                r = self.rec_coo(nb_rec, prevent_overlap=True)
                i1, j1, i2, j2 = r[nb_rec - 1]
                if i1 + 3 < i2 and j1 + 3 < j2:
                    break

            for n in range(nb_rec):
                i1, j1, i2, j2 = r[n]
                if n == nb_rec - 1:
                    if direction == 0:
                        X[i1 + 1 : i2 - 1, j1 + 1 : j2 - 1] = c[n]
                        f_X[i1:i2, j1:j2] = c[n]
                    else:
                        X[i1:i2, j1:j2] = c[n]
                        f_X[i1 + 1 : i2 - 1, j1 + 1 : j2 - 1] = c[n]
                else:
                    X[i1:i2, j1:j2] = c[n]
                    f_X[i1:i2, j1:j2] = c[n]

    # @torch.compile
    def task_half_fill(self, A, f_A, B, f_B):
        di, dj = torch.randint(2, (2,)) * 2 - 1
        nb_rec = 3
        c = torch.randperm(len(self.colors) - 1)[: 2 * nb_rec] + 1
        direction = torch.randint(4, (1,)).item()
        for X, f_X in [(A, f_A), (B, f_B)]:
            r = self.rec_coo(nb_rec, prevent_overlap=True)
            for n in range(nb_rec):
                i1, j1, i2, j2 = r[n]
                X[i1:i2, j1:j2] = c[2 * n]
                f_X[i1:i2, j1:j2] = c[2 * n]
                # Not my proudest moment
                if direction == 0:
                    i = (i1 + i2) // 2
                    X[i : i + 1, j1:j2] = c[2 * n + 1]
                    if n == nb_rec - 1:
                        f_X[i:i2, j1:j2] = c[2 * n + 1]
                    else:
                        f_X[i : i + 1, j1:j2] = c[2 * n + 1]
                elif direction == 1:
                    i = (i1 + i2 - 1) // 2
                    X[i : i + 1, j1:j2] = c[2 * n + 1]
                    if n == nb_rec - 1:
                        f_X[i1 : i + 1, j1:j2] = c[2 * n + 1]
                    else:
                        f_X[i : i + 1, j1:j2] = c[2 * n + 1]
                elif direction == 2:
                    j = (j1 + j2) // 2
                    X[i1:i2, j : j + 1] = c[2 * n + 1]
                    if n == nb_rec - 1:
                        f_X[i1:i2, j:j2] = c[2 * n + 1]
                    else:
                        f_X[i1:i2, j : j + 1] = c[2 * n + 1]
                elif direction == 3:
                    j = (j1 + j2 - 1) // 2
                    X[i1:i2, j : j + 1] = c[2 * n + 1]
                    if n == nb_rec - 1:
                        f_X[i1:i2, j1 : j + 1] = c[2 * n + 1]
                    else:
                        f_X[i1:i2, j : j + 1] = c[2 * n + 1]

    # @torch.compile
    def task_frame(self, A, f_A, B, f_B):
        nb_rec = 3
        c = torch.randperm(len(self.colors) - 1)[: nb_rec + 1] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            r = self.rec_coo(nb_rec, prevent_overlap=True)
            for n in range(nb_rec):
                i1, j1, i2, j2 = r[n]
                X[i1:i2, j1:j2] = c[n]
                if n == nb_rec - 1:
                    f_X[i1:i2, j1] = c[n]
                    f_X[i1:i2, j2 - 1] = c[n]
                    f_X[i1, j1:j2] = c[n]
                    f_X[i2 - 1, j1:j2] = c[n]
                else:
                    f_X[i1:i2, j1:j2] = c[n]

    # @torch.compile
    def task_detect(self, A, f_A, B, f_B):
        nb_rec = 3
        c = torch.randperm(len(self.colors) - 1)[: nb_rec + 1] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            r = self.rec_coo(nb_rec, prevent_overlap=True)
            for n in range(nb_rec):
                i1, j1, i2, j2 = r[n]
                X[i1:i2, j1:j2] = c[n]
                if n < nb_rec - 1:
                    f_X[i1, j1] = c[-1]

    # @torch.compile
    def contact(self, X, i, j, q):
        nq, nq_diag = 0, 0
        no = 0

        for ii, jj in [
            (i - 1, j - 1),
            (i - 1, j),
            (i - 1, j + 1),
            (i, j - 1),
            (i, j + 1),
            (i + 1, j - 1),
            (i + 1, j),
            (i + 1, j + 1),
        ]:
            if ii >= 0 and ii < self.height and jj >= 0 and jj < self.width:
                if X[ii, jj] != 0 and X[ii, jj] != q:
                    no += 1

        for ii, jj in [
            (i - 1, j - 1),
            (i - 1, j + 1),
            (i + 1, j - 1),
            (i + 1, j + 1),
        ]:
            if ii >= 0 and ii < self.height and jj >= 0 and jj < self.width:
                if X[ii, jj] == q and X[i, jj] != q and X[ii, j] != q:
                    nq_diag += 1

        for ii, jj in [(i - 1, j), (i, j - 1), (i, j + 1), (i + 1, j)]:
            if ii >= 0 and ii < self.height and jj >= 0 and jj < self.width:
                if X[ii, jj] == q:
                    nq += 1

        return no, nq, nq_diag

    def task_count(self, A, f_A, B, f_B):
        while True:
            error = False

            N = torch.randint(5, (1,)).item() + 2
            c = torch.zeros(N + 1, dtype=torch.int64)
            c[1:] = torch.randperm(len(self.colors) - 1)[:N] + 1

            for X, f_X in [(A, f_A), (B, f_B)]:
                if not hasattr(self, "cache_count") or len(self.cache_count) == 0:
                    self.cache_count = list(
                        grow_islands(
                            1000,
                            self.height,
                            self.width,
                            nb_seeds=self.height * self.width // 8,
                            nb_iterations=self.height * self.width // 10,
                        )
                    )

                X[...] = self.cache_count.pop()

                # k = (X.max() + 1 + (c.size(0) - 1)).item()
                # V = torch.arange(k) // (c.size(0) - 1)
                # V = (V + torch.rand(V.size())).sort().indices[: X.max() + 1] % (
                # c.size(0) - 1
                # ) + 1
                V = torch.randint(c.size(0) - 1, (X.max() + 1,)) + 1
                V[0] = 0
                NB = F.one_hot(c[V]).sum(dim=0)
                X[...] = c[V[X]]

                if F.one_hot(X.flatten()).max(dim=0).values.sum().item() == N + 1:
                    f_X[...] = 0
                    for e in range(1, N + 1):
                        for j in range(NB[c[e]]):
                            if j < self.width:
                                f_X[e - 1, j] = c[e]
                            else:
                                error = True
                                break
                else:
                    error = True
                    break

            if not error:
                break

        assert F.one_hot(A.flatten()).max(dim=0).values.sum() >= 3

    # @torch.compile
    def task_trajectory(self, A, f_A, B, f_B):
        c = torch.randperm(len(self.colors) - 1)[:2] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            while True:
                di, dj = torch.randint(7, (2,)) - 3
                i, j = (
                    torch.randint(self.height, (1,)).item(),
                    torch.randint(self.width, (1,)).item(),
                )
                if (
                    abs(di) + abs(dj) > 0
                    and i + 2 * di >= 0
                    and i + 2 * di < self.height
                    and j + 2 * dj >= 0
                    and j + 2 * dj < self.width
                ):
                    break

            k = 0
            while (
                i + k * di >= 0
                and i + k * di < self.height
                and j + k * dj >= 0
                and j + k * dj < self.width
            ):
                if k < 2:
                    X[i + k * di, j + k * dj] = c[k]
                f_X[i + k * di, j + k * dj] = c[min(k, 1)]
                k += 1

    # @torch.compile
    def task_bounce(self, A, f_A, B, f_B):
        c = torch.randperm(len(self.colors) - 1)[:3] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            # @torch.compile
            def free(i, j):
                return (
                    i >= 0
                    and i < self.height
                    and j >= 0
                    and j < self.width
                    and f_X[i, j] == 0
                )

            while True:
                f_X[...] = 0
                X[...] = 0

                for _ in range((self.height * self.width) // 10):
                    i, j = (
                        torch.randint(self.height, (1,)).item(),
                        torch.randint(self.width, (1,)).item(),
                    )
                    X[i, j] = c[0]
                    f_X[i, j] = c[0]

                while True:
                    di, dj = torch.randint(7, (2,)) - 3
                    if abs(di) + abs(dj) == 1:
                        break

                i, j = (
                    torch.randint(self.height, (1,)).item(),
                    torch.randint(self.width, (1,)).item(),
                )

                X[i, j] = c[1]
                f_X[i, j] = c[1]
                l = 0

                while True:
                    l += 1
                    if free(i + di, j + dj):
                        pass
                    elif free(i - dj, j + di):
                        di, dj = -dj, di
                        if free(i + dj, j - di):
                            if torch.rand(1) < 0.5:
                                di, dj = -di, -dj
                    elif free(i + dj, j - di):
                        di, dj = dj, -di
                    else:
                        break

                    i, j = i + di, j + dj
                    f_X[i, j] = c[2]
                    if l <= 1:
                        X[i, j] = c[2]

                    if l >= self.width:
                        break

                f_X[i, j] = c[1]
                X[i, j] = c[1]

                if l > 3:
                    break

    # @torch.compile
    def task_scale(self, A, f_A, B, f_B):
        c = torch.randperm(len(self.colors) - 1)[:2] + 1

        i, j = (
            torch.randint(self.height // 2, (1,)).item(),
            torch.randint(self.width // 2, (1,)).item(),
        )

        for X, f_X in [(A, f_A), (B, f_B)]:
            for _ in range(3):
                while True:
                    i1, j1 = (
                        torch.randint(self.height // 2 + 1, (1,)).item(),
                        torch.randint(self.width // 2 + 1, (1,)).item(),
                    )
                    i2, j2 = (
                        torch.randint(self.height // 2 + 1, (1,)).item(),
                        torch.randint(self.width // 2 + 1, (1,)).item(),
                    )
                    if i1 < i2 and j1 < j2 and min(i2 - i1, j2 - j1) <= 3:
                        break
                X[i + i1 : i + i2, j + j1 : j + j2] = c[0]
                f_X[2 * i1 : 2 * i2, 2 * j1 : 2 * j2] = c[0]

            X[i, j] = c[1]
            f_X[0:2, 0:2] = c[1]

    # @torch.compile
    def task_symbols(self, A, f_A, B, f_B):
        nb_rec = 4
        c = torch.randperm(len(self.colors) - 1)[: nb_rec + 1] + 1
        delta = 3
        for X, f_X in [(A, f_A), (B, f_B)]:
            while True:
                i, j = torch.randint(self.height - delta + 1, (nb_rec,)), torch.randint(
                    self.width - delta + 1, (nb_rec,)
                )
                d = (i[None, :] - i[:, None]).abs().max((j[None, :] - j[:, None]).abs())
                d.fill_diagonal_(delta + 1)
                if d.min() > delta:
                    break

            for k in range(1, nb_rec):
                X[i[k] : i[k] + delta, j[k] : j[k] + delta] = c[k]

            ai, aj = i.float().mean(), j.float().mean()

            q = torch.randint(3, (1,)).item() + 1

            X[i[0] + delta // 2 - 1, j[0] + delta // 2 - 1] = c[0]
            X[i[0] + delta // 2 - 1, j[0] + delta // 2 + 1] = c[0]
            X[i[0] + delta // 2 + 1, j[0] + delta // 2 - 1] = c[0]
            X[i[0] + delta // 2 + 1, j[0] + delta // 2 + 1] = c[0]

            assert i[q] != ai and j[q] != aj

            X[
                i[0] + delta // 2 + (i[q] - ai).sign().long(),
                j[0] + delta // 2 + (j[q] - aj).sign().long(),
            ] = c[nb_rec]

            f_X[i[0] : i[0] + delta, j[0] : j[0] + delta] = c[q]

    # @torch.compile
    def task_isometry(self, A, f_A, B, f_B):
        nb_rec = 3
        di, dj = torch.randint(3, (2,)) - 1
        o = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        m = torch.eye(2)
        for _ in range(torch.randint(4, (1,)).item()):
            m = m @ o
        if torch.rand(1) < 0.5:
            m[0, :] = -m[0, :]

        ci, cj = (self.height - 1) / 2, (self.width - 1) / 2

        for X, f_X in [(A, f_A), (B, f_B)]:
            while True:
                X[...] = 0
                f_X[...] = 0

                c = torch.randperm(len(self.colors) - 1)[:nb_rec] + 1

                for r in range(nb_rec):
                    while True:
                        i1, i2 = torch.randint(self.height - 2, (2,)) + 1
                        j1, j2 = torch.randint(self.width - 2, (2,)) + 1
                        if (
                            i2 >= i1
                            and j2 >= j1
                            and max(i2 - i1, j2 - j1) >= 2
                            and min(i2 - i1, j2 - j1) <= 3
                        ):
                            break
                    X[i1 : i2 + 1, j1 : j2 + 1] = c[r]

                    i1, j1, i2, j2 = i1 - ci, j1 - cj, i2 - ci, j2 - cj

                    i1, j1 = m[0, 0] * i1 + m[0, 1] * j1, m[1, 0] * i1 + m[1, 1] * j1
                    i2, j2 = m[0, 0] * i2 + m[0, 1] * j2, m[1, 0] * i2 + m[1, 1] * j2

                    i1, j1, i2, j2 = i1 + ci, j1 + cj, i2 + ci, j2 + cj
                    i1, i2 = i1.long() + di, i2.long() + di
                    j1, j2 = j1.long() + dj, j2.long() + dj
                    if i1 > i2:
                        i1, i2 = i2, i1
                    if j1 > j2:
                        j1, j2 = j2, j1

                    f_X[i1 : i2 + 1, j1 : j2 + 1] = c[r]

                n = F.one_hot(X.flatten()).sum(dim=0)[1:]
                if (
                    n.sum() > self.height * self.width // 4
                    and (n > 0).long().sum() == nb_rec
                ):
                    break

    def compute_distance(self, walls, goal_i, goal_j):
        max_length = walls.numel()
        dist = torch.full_like(walls, max_length)

        dist[goal_i, goal_j] = 0
        pred_dist = torch.empty_like(dist)

        while True:
            pred_dist.copy_(dist)
            dist[1:-1, 1:-1] = (
                torch.cat(
                    (
                        dist[None, 1:-1, 1:-1],
                        dist[None, 1:-1, 0:-2],
                        dist[None, 2:, 1:-1],
                        dist[None, 1:-1, 2:],
                        dist[None, 0:-2, 1:-1],
                    ),
                    0,
                ).min(dim=0)[0]
                + 1
            )

            dist = walls * max_length + (1 - walls) * dist

            if dist.equal(pred_dist):
                return dist * (1 - walls)

    # @torch.compile
    def task_distance(self, A, f_A, B, f_B):
        c = torch.randperm(len(self.colors) - 1)[:3] + 1
        dist0 = torch.empty(self.height + 2, self.width + 2)
        dist1 = torch.empty(self.height + 2, self.width + 2)
        for X, f_X in [(A, f_A), (B, f_B)]:
            nb_rec = torch.randint(3, (1,)).item() + 1
            while True:
                r = self.rec_coo(nb_rec, prevent_overlap=True)
                X[...] = 0
                f_X[...] = 0
                for n in range(nb_rec):
                    i1, j1, i2, j2 = r[n]
                    X[i1:i2, j1:j2] = c[0]
                    f_X[i1:i2, j1:j2] = c[0]
                while True:
                    i0, j0 = (
                        torch.randint(self.height, (1,)).item(),
                        torch.randint(self.width, (1,)).item(),
                    )
                    if X[i0, j0] == 0:
                        break
                while True:
                    i1, j1 = (
                        torch.randint(self.height, (1,)).item(),
                        torch.randint(self.width, (1,)).item(),
                    )
                    if X[i1, j1] == 0:
                        break
                dist1[...] = 1
                dist1[1:-1, 1:-1] = (X != 0).long()
                dist1[...] = self.compute_distance(dist1, i1 + 1, j1 + 1)
                if (
                    dist1[i0 + 1, j0 + 1] >= 1
                    and dist1[i0 + 1, j0 + 1] < self.height * 4
                ):
                    break

            dist0[...] = 1
            dist0[1:-1, 1:-1] = (X != 0).long()
            dist0[...] = self.compute_distance(dist0, i0 + 1, j0 + 1)

            dist0 = dist0[1:-1, 1:-1]
            dist1 = dist1[1:-1, 1:-1]

            D = dist1[i0, j0]
            for d in range(1, D):
                M = (dist0 == d) & (dist1 == D - d)
                f_X[...] = (1 - M) * f_X + M * c[1]

            X[i0, j0] = c[2]
            f_X[i0, j0] = c[2]
            X[i1, j1] = c[2]
            f_X[i1, j1] = c[2]

    # for X, f_X in [(A, f_A), (B, f_B)]:
    # n = torch.arange(self.height * self.width).reshape(self.height, self.width)
    # k = torch.randperm(self.height * self.width)
    # X[...]=-1
    # for q in k:
    # i,j=q%self.height,q//self.height
    # if

    # @torch.compile
    def task_puzzle(self, A, f_A, B, f_B):
        S = 4
        i0, j0 = (self.height - S) // 2, (self.width - S) // 2
        c = torch.randperm(len(self.colors) - 1)[:4] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            while True:
                f_X[...] = 0
                h = list(torch.randperm(c.size(0)))
                n = torch.zeros(c.max() + 1)
                for _ in range(2):
                    k = torch.randperm(S * S)
                    for q in k:
                        i, j = q % S + i0, q // S + j0
                        if f_X[i, j] == 0:
                            r, s, t, u = (
                                f_X[i - 1, j],
                                f_X[i, j - 1],
                                f_X[i + 1, j],
                                f_X[i, j + 1],
                            )
                            r, s, t, u = torch.tensor([r, s, t, u])[torch.randperm(4)]
                            if r > 0 and n[r] < 6:
                                n[r] += 1
                                f_X[i, j] = r
                            elif s > 0 and n[s] < 6:
                                n[s] += 1
                                f_X[i, j] = s
                            elif t > 0 and n[t] < 6:
                                n[t] += 1
                                f_X[i, j] = t
                            elif u > 0 and n[u] < 6:
                                n[u] += 1
                                f_X[i, j] = u
                            else:
                                if len(h) > 0:
                                    d = c[h.pop()]
                                    n[d] += 1
                                    f_X[i, j] = d

                if n.sum() == S * S:
                    break

            k = 0
            for d in range(4):
                while True:
                    ii, jj = (
                        torch.randint(self.height, (1,)).item(),
                        torch.randint(self.width, (1,)).item(),
                    )
                    e = 0
                    for i in range(S):
                        for j in range(S):
                            if (
                                ii + i >= self.height
                                or jj + j >= self.width
                                or (
                                    f_X[i + i0, j + j0] == c[d]
                                    and X[ii + i, jj + j] > 0
                                )
                            ):
                                e = 1
                    if e == 0:
                        break
                for i in range(S):
                    for j in range(S):
                        if f_X[i + i0, j + j0] == c[d]:
                            X[ii + i, jj + j] = c[d]

    def task_islands(self, A, f_A, B, f_B):
        c = torch.randperm(len(self.colors) - 1)[:2] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            if not hasattr(self, "cache_islands") or len(self.cache_islands) == 0:
                self.cache_islands = list(
                    grow_islands(
                        1000,
                        self.height,
                        self.width,
                        nb_seeds=self.height * self.width // 20,
                        nb_iterations=self.height * self.width // 2,
                    )
                )

            A = self.cache_islands.pop()

            while True:
                i, j = (
                    torch.randint(self.height // 2, (1,)).item(),
                    torch.randint(self.width // 2, (1,)).item(),
                )
                if A[i, j] > 0:
                    break

            X[...] = (A > 0) * c[0]
            X[i, j] = c[1]
            f_X[...] = (A == A[i, j]) * c[1] + ((A > 0) & (A != A[i, j])) * c[0]

    # @torch.compile
    def task_stack(self, A, f_A, B, f_B):
        N = 5
        c = torch.randperm(len(self.colors) - 1)[:N] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            i1, j1, i2, j2 = (
                self.height // 2 - 1,
                self.width // 2 - 1,
                self.height // 2 + 1,
                self.width // 2 + 1,
            )
            op = torch.tensor((0, 1, 2, 3) * 4)
            op = op[torch.randperm(op.size(0))[:9]]
            for q in range(op.size(0)):
                u = 3 * (q // 3)
                v = 3 * (q % 3)
                d = c[torch.randint(N, (1,)).item()]
                # X[u+1,v+1]=d
                if op[q] == 0:  # right
                    X[u : u + 3, v + 2] = d
                elif op[q] == 1:  # let
                    X[u : u + 3, v] = d
                elif op[q] == 2:  # bottom
                    X[u + 2, v : v + 3] = d
                elif op[q] == 3:  # top
                    X[u, v : v + 3] = d

                if q == 0:
                    f_X[i1:i2, j1:j2] = d
                elif op[q] == 0:  # right
                    f_X[i1:i2, j2] = d
                    j2 += 1
                elif op[q] == 1:  # let
                    j1 -= 1
                    f_X[i1:i2, j1] = d
                elif op[q] == 2:  # bottom
                    f_X[i2, j1:j2] = d
                    i2 += 1
                elif op[q] == 3:  # top
                    i1 -= 1
                    f_X[i1, j1:j2] = d

    def randint(self, *m):
        m = torch.tensor(m)
        return (torch.rand(m.size()) * m).long()

    def task_matrices(self, A, f_A, B, f_B):
        N = 6
        c = torch.randperm(len(self.colors) - 1)[:N] + 1

        for X, f_X in [(A, f_A), (B, f_B)]:
            M1 = torch.randint(2, (5, 5))
            M2 = torch.randint(2, (5, 5))
            P = M1 @ M2
            for i in range(5):
                for j in range(5):
                    X[i, j] = c[M1[i, j]]
                    X[i, j + 5] = c[M2[i, j]]
                    f_X[i, j] = c[M1[i, j]]
                    f_X[i, j + 5] = c[M2[i, j]]
                    f_X[i + 5, j + 5] = c[P[i, j]]

    def task_compute(self, A, f_A, B, f_B):
        N = 6
        c = torch.randperm(len(self.colors) - 1)[:N] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            v = torch.randint((self.width - 1) // 2, (N,)) + 1
            chain = torch.randperm(N)
            eq = []
            for i in range(chain.size(0) - 1):
                i1, i2 = chain[i], chain[i + 1]
                v1, v2 = v[i1], v[i2]
                k = torch.arange(self.width // 2) + 1
                d = ((k[None, :] * v1 - k[:, None] * v2) == 0).nonzero() + 1
                d = d[torch.randint(d.size(0), (1,)).item()]
                w1, w2 = d
                eq.append((c[i1], w1, c[i2], w2))

            ii = torch.randperm(self.height - 2)[: len(eq)]

            for k, x in enumerate(eq):
                i = ii[k]
                c1, w1, c2, w2 = x
                s = torch.randint(self.width - (w1 + w2) + 1, (1,)).item()
                X[i, s : s + w1] = c1
                X[i, s + w1 : s + w1 + w2] = c2
                f_X[i, s : s + w1] = c1
                f_X[i, s + w1 : s + w1 + w2] = c2

            i1, i2 = torch.randperm(N)[:2]
            v1, v2 = v[i1], v[i2]
            k = torch.arange(self.width // 2) + 1
            d = ((k[None, :] * v1 - k[:, None] * v2) == 0).nonzero() + 1
            d = d[torch.randint(d.size(0), (1,)).item()]
            w1, w2 = d
            c1, c2 = c[i1], c[i2]
            s = 0  # torch.randint(self.width - (w1 + w2) + 1, (1,)).item()
            i = self.height - 1
            X[i, s : s + w1] = c1
            X[i, s + w1 : s + w1 + 1] = c2
            f_X[i, s : s + w1] = c1
            f_X[i, s + w1 : s + w1 + w2] = c2

    ######################################################################

    def trivial_prompts_and_answers(self, prompts, answers):
        S = self.height * self.width
        Bs = prompts[:, 2 * (S + 1) : 2 * (S + 1) + S]
        f_Bs = answers
        return (Bs == f_Bs).long().min(dim=-1).values > 0

    def generate_prompts_and_answers_(self, nb, tasks=None, progress_bar=False):
        if tasks is None:
            tasks = self.all_tasks

        S = self.height * self.width
        prompts = torch.zeros(nb, 3 * S + 2, dtype=torch.int64)
        answers = torch.zeros(nb, S, dtype=torch.int64)

        bunch = zip(prompts, answers)

        if progress_bar:
            bunch = tqdm.tqdm(
                bunch,
                dynamic_ncols=True,
                desc="world generation",
                total=prompts.size(0),
            )

        for prompt, answer in bunch:
            A = prompt[0 * (S + 1) : 0 * (S + 1) + S].view(self.height, self.width)
            f_A = prompt[1 * (S + 1) : 1 * (S + 1) + S].view(self.height, self.width)
            B = prompt[2 * (S + 1) : 2 * (S + 1) + S].view(self.height, self.width)
            f_B = answer.view(self.height, self.width)
            task = tasks[torch.randint(len(tasks), (1,)).item()]
            task(A, f_A, B, f_B)

        return prompts.flatten(1), answers.flatten(1)

    def save_quiz_illustrations(
        self,
        result_dir,
        filename_prefix,
        prompts,
        answers,
        predicted_prompts=None,
        predicted_answers=None,
        nrow=4,
    ):
        self.save_image(
            result_dir,
            filename_prefix + ".png",
            prompts,
            answers,
            predicted_prompts,
            predicted_answers,
            nrow,
        )

    def save_some_examples(self, result_dir):
        nb, nrow = 128, 4
        for t in self.all_tasks:
            print(t.__name__)
            prompts, answers = self.generate_prompts_and_answers_(nb, tasks=[t])
            self.save_quiz_illustrations(
                result_dir, t.__name__, prompts[:nb], answers[:nb], nrow=nrow
            )


######################################################################

if __name__ == "__main__":
    import time

    # grids = Grids(max_nb_cached_chunks=5, chunk_size=100, nb_threads=4)
    grids = Grids()

    # nb = 1000
    # grids = problem.MultiThreadProblem(
    # grids, max_nb_cached_chunks=50, chunk_size=100, nb_threads=1
    # )
    #    time.sleep(10)
    # start_time = time.perf_counter()
    # prompts, answers = grids.generate_prompts_and_answers(nb)
    # delay = time.perf_counter() - start_time
    # print(f"{prompts.size(0)/delay:02f} seq/s")
    # exit(0)

    # if True:
    nb, nrow = 128, 4
    # nb, nrow = 8, 2

    # for t in grids.all_tasks:
    for t in [grids.task_replace_color]:
        print(t.__name__)
        prompts, answers = grids.generate_prompts_and_answers_(nb, tasks=[t])
        grids.save_quiz_illustrations(
            "/tmp", t.__name__, prompts[:nb], answers[:nb], nrow=nrow
        )

    # exit(0)

    nb = 1000

    # for t in grids.all_tasks:
    for t in [grids.task_compute]:
        start_time = time.perf_counter()
        prompts, answers = grids.generate_prompts_and_answers_(nb, tasks=[t])
        delay = time.perf_counter() - start_time
        print(f"{t.__name__} {prompts.size(0)/delay:02f} seq/s")

    exit(0)

    m = torch.randint(2, (prompts.size(0),))
    predicted_prompts = m * (torch.randint(2, (prompts.size(0),)) * 2 - 1)
    predicted_answers = (1 - m) * (torch.randint(2, (prompts.size(0),)) * 2 - 1)

    grids.save_quiz_illustrations(
        "/tmp",
        "test",
        prompts[:nb],
        answers[:nb],
        # You can add a bool to put a frame around the predicted parts
        predicted_prompts[:nb],
        predicted_answers[:nb],
    )
