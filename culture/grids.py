#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import math, sys, tqdm, os, warnings, cairo

import torch, torchvision

from torch import nn
from torch.nn import functional as F

######################################################################


def text_img(height, width, text):
    pixel_map = torch.full((height, width, 4), 255, dtype=torch.uint8)

    surface = cairo.ImageSurface.create_for_data(
        pixel_map.numpy(), cairo.FORMAT_ARGB32, pixel_map.size(1), pixel_map.size(0)
    )

    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_font_size(16)
    ctx.select_font_face("courier", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    y = None
    for line in text.split("\n"):
        xbearing, ybearing, width, height, dx, dy = ctx.text_extents(line)
        if y is None:
            y = height * 1.5
            x = height * 0.5

        ctx.move_to(x, y)
        ctx.show_text(line)
        y += height * 1.5

    ctx.stroke()

    return pixel_map.permute(2, 0, 1)[None, :3].contiguous()


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

    def check_structure(self, quizzes, struct):
        S = self.height * self.width

        return (
            (quizzes[:, 0 * (S + 1)] == self.l2tok[struct[0]])
            & (quizzes[:, 1 * (S + 1)] == self.l2tok[struct[1]])
            & (quizzes[:, 2 * (S + 1)] == self.l2tok[struct[2]])
            & (quizzes[:, 3 * (S + 1)] == self.l2tok[struct[3]])
        ).all()

    def get_structure(self, quizzes):
        S = self.height * self.width
        struct = tuple(
            self.tok2l[n.item()]
            for n in quizzes.reshape(quizzes.size(0), 4, S + 1)[0, :, 0]
        )
        self.check_structure(quizzes, struct)
        return struct

    def inject_noise(self, quizzes, noise, struct, mask):
        assert self.check_structure(quizzes, struct=struct)
        S = self.height * self.width

        mask = torch.tensor(mask, device=quizzes.device)
        mask = mask[None, :, None].expand(1, 4, S + 1).clone()
        mask[:, :, 0] = 0
        mask = mask.reshape(1, -1).expand_as(quizzes)
        mask = mask * (torch.rand(mask.size(), device=mask.device) <= noise).long()
        random = torch.randint(self.nb_colors, mask.size())
        quizzes = mask * random + (1 - mask) * quizzes

        return quizzes

    # What a mess
    def reconfigure(self, quizzes, struct=("A", "f_A", "B", "f_B")):
        if torch.is_tensor(quizzes):
            return self.reconfigure([quizzes], struct=struct)[0]

        S = self.height * self.width
        result = [x.new(x.size()) for x in quizzes]

        struct_from = self.get_structure(quizzes[0][:1])
        i = self.indices_select(quizzes[0], struct_from)

        sf = dict((l, n) for n, l in enumerate(struct_from))

        for q in range(4):
            k = sf[struct[q]]
            for x, y in zip(quizzes, result):
                l = x.size(1) // 4
                y[i, q * l : (q + 1) * l] = x[i, k * l : (k + 1) * l]

        j = i == False

        if j.any():
            for z, y in zip(
                self.reconfigure([x[j] for x in quizzes], struct=struct), result
            ):
                y[j] = z

        return result

    def trivial(self, quizzes):
        S = self.height * self.width
        assert self.check_structure(quizzes, struct=("A", "f_A", "B", "f_B"))
        a = quizzes.reshape(quizzes.size(0), 4, S + 1)[:, :, 1:]
        return (a[:, 0] == a[:, 1]).min(dim=1).values | (a[:, 2] == a[:, 3]).min(
            dim=1
        ).values

    def make_ar_mask(self, quizzes, struct=("A", "f_A", "B", "f_B"), mask=(0, 0, 0, 1)):
        assert self.check_structure(quizzes, struct)

        ar_mask = quizzes.new_zeros(quizzes.size())

        S = self.height * self.width
        a = ar_mask.reshape(ar_mask.size(0), 4, S + 1)[:, :, 1:]
        a[:, 0, :] = mask[0]
        a[:, 1, :] = mask[1]
        a[:, 2, :] = mask[2]
        a[:, 3, :] = mask[3]

        return ar_mask

    def indices_select(self, quizzes, struct=("A", "f_A", "B", "f_B")):
        S = self.height * self.width
        q = quizzes.reshape(quizzes.size(0), 4, S + 1)
        return (
            (q[:, 0, 0] == self.l2tok[struct[0]])
            & (q[:, 1, 0] == self.l2tok[struct[1]])
            & (q[:, 2, 0] == self.l2tok[struct[2]])
            & (q[:, 3, 0] == self.l2tok[struct[3]])
        )

    def __init__(
        self,
        max_nb_cached_chunks=None,
        chunk_size=None,
        nb_threads=-1,
        tasks=None,
    ):
        self.colors = torch.tensor([c for _, c in self.named_colors])

        self.nb_colors = len(self.colors)
        self.token_A = self.nb_colors
        self.token_f_A = self.token_A + 1
        self.token_B = self.token_f_A + 1
        self.token_f_B = self.token_B + 1

        self.nb_rec_max = 5
        self.rfree = torch.tensor([])

        self.l2tok = {
            "A": self.token_A,
            "f_A": self.token_f_A,
            "B": self.token_B,
            "f_B": self.token_f_B,
        }

        self.tok2l = {
            self.token_A: "A",
            self.token_f_A: "f_A",
            self.token_B: "B",
            self.token_f_B: "f_B",
        }

        self.height = 10
        self.width = 10
        self.seq_len = 4 * (1 + self.height * self.width)
        self.nb_token_values = self.token_f_B + 1

        self.cache_rec_coo = {}

        all_tasks = [
            self.task_replace_color,
            self.task_translate,
            self.task_grow,
            self.task_half_fill,
            self.task_frame,
            self.task_detect,
            self.task_scale,
            self.task_symbols,
            self.task_corners,
            self.task_contact,
            self.task_path,
            self.task_fill,
            ############################################ hard ones
            self.task_isometry,
            self.task_trajectory,
            self.task_bounce,
            # self.task_count, # NOT REVERSIBLE
            # self.task_islands, # TOO MESSY
        ]

        if tasks is None:
            self.all_tasks = all_tasks
        else:
            self.all_tasks = [getattr(self, "task_" + t) for t in tasks.split(",")]

        super().__init__(max_nb_cached_chunks, chunk_size, nb_threads)

    ######################################################################

    def grid2img(self, x, scale=15):
        m = torch.logical_and(x >= 0, x < self.nb_colors).long()
        y = self.colors[x * m].permute(0, 3, 1, 2)
        s = y.shape
        y = y[:, :, :, None, :, None].expand(-1, -1, -1, scale, -1, scale)
        y = y.reshape(s[0], s[1], s[2] * scale, s[3] * scale)

        y[:, :, :, torch.arange(0, y.size(3), scale)] = 64
        y[:, :, torch.arange(0, y.size(2), scale), :] = 64

        for n in range(m.size(0)):
            for i in range(m.size(1)):
                for j in range(m.size(2)):
                    if m[n, i, j] == 0:
                        for k in range(3, scale - 2):
                            y[n, :, i * scale + k, j * scale + k] = 0
                            y[n, :, i * scale + k, j * scale + scale - k] = 0

        y = y[:, :, 1:, 1:]

        return y

    def add_frame(self, img, colors, thickness):
        result = img.new(
            img.size(0),
            img.size(1),
            img.size(2) + 2 * thickness,
            img.size(3) + 2 * thickness,
        )

        result[...] = colors[:, :, None, None]
        result[:, :, thickness:-thickness, thickness:-thickness] = img

        return result

    def save_quizzes_as_image(
        self,
        result_dir,
        filename,
        quizzes,
        predicted_parts=None,
        correct_parts=None,
        comments=None,
        comment_height=48,
        nrow=4,
        margin=8,
    ):
        quizzes = quizzes.to("cpu")

        to_reconfigure = [quizzes]
        if predicted_parts is not None:
            to_reconfigure.append(predicted_parts)
        if correct_parts is not None:
            to_reconfigure.append(correct_parts)

        to_reconfigure = self.reconfigure(to_reconfigure, ("A", "f_A", "B", "f_B"))

        quizzes = to_reconfigure.pop(0)
        if predicted_parts is not None:
            predicted_parts = to_reconfigure.pop(0)
        if correct_parts is not None:
            correct_parts = to_reconfigure.pop(0)

        S = self.height * self.width

        A, f_A, B, f_B = (
            quizzes.reshape(quizzes.size(0), 4, S + 1)[:, :, 1:]
            .reshape(quizzes.size(0), 4, self.height, self.width)
            .permute(1, 0, 2, 3)
        )

        frame, white, gray, green, red = torch.tensor(
            [[64, 64, 64], [255, 255, 255], [200, 200, 200], [0, 255, 0], [255, 0, 0]],
            device=quizzes.device,
        )

        img_A = self.add_frame(self.grid2img(A), frame[None, :], thickness=1)
        img_f_A = self.add_frame(self.grid2img(f_A), frame[None, :], thickness=1)
        img_B = self.add_frame(self.grid2img(B), frame[None, :], thickness=1)
        img_f_B = self.add_frame(self.grid2img(f_B), frame[None, :], thickness=1)

        # predicted_parts Nx4
        # correct_parts Nx4

        if predicted_parts is None:
            colors = white[None, None, :].expand(-1, 4, -1)
        else:
            predicted_parts = predicted_parts.to("cpu")
            if correct_parts is None:
                colors = (
                    predicted_parts[:, :, None] * gray[None, None, :]
                    + (1 - predicted_parts[:, :, None]) * white[None, None, :]
                )
            else:
                correct_parts = correct_parts.to("cpu")
                colors = (
                    predicted_parts[:, :, None]
                    * (
                        (correct_parts[:, :, None] == 1).long() * green[None, None, :]
                        + (correct_parts[:, :, None] == 0).long() * gray[None, None, :]
                        + (correct_parts[:, :, None] == -1).long() * red[None, None, :]
                    )
                    + (1 - predicted_parts[:, :, None]) * white[None, None, :]
                )

        img_A = self.add_frame(img_A, colors[:, 0], thickness=8)
        img_f_A = self.add_frame(img_f_A, colors[:, 1], thickness=8)
        img_B = self.add_frame(img_B, colors[:, 2], thickness=8)
        img_f_B = self.add_frame(img_f_B, colors[:, 3], thickness=8)

        img_A = self.add_frame(img_A, white[None, :], thickness=2)
        img_f_A = self.add_frame(img_f_A, white[None, :], thickness=2)
        img_B = self.add_frame(img_B, white[None, :], thickness=2)
        img_f_B = self.add_frame(img_f_B, white[None, :], thickness=2)

        img = torch.cat([img_A, img_f_A, img_B, img_f_B], dim=3)

        if comments is not None:
            comment_img = [text_img(comment_height, img.size(3), t) for t in comments]
            comment_img = torch.cat(comment_img, dim=0)
            img = torch.cat([img, comment_img], dim=2)

        image_name = os.path.join(result_dir, filename)

        torchvision.utils.save_image(
            img.float() / 255.0,
            image_name,
            nrow=nrow,
            padding=margin * 4,
            pad_value=1.0,
        )

    ######################################################################

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
                i[:, 1] += 1
                j[:, 1] += 1
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
                    no_overlap = (
                        (A_i1 >= B_i2)
                        | (A_i2 <= B_i1)
                        | (A_j1 >= B_j2)
                        | (A_j2 <= B_j1)
                    )
                    i, j = (i[no_overlap], j[no_overlap])
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

    def contact_matrices(self, rn, ri, rj, rz):
        n = torch.arange(self.nb_rec_max)
        return (
            (
                (
                    (
                        (ri[:, :, None, 0] == ri[:, None, :, 1] + 1)
                        | (ri[:, :, None, 1] + 1 == ri[:, None, :, 0])
                    )
                    & (rj[:, :, None, 0] <= rj[:, None, :, 1])
                    & (rj[:, :, None, 1] >= rj[:, None, :, 0])
                )
                | (
                    (
                        (rj[:, :, None, 0] == rj[:, None, :, 1] + 1)
                        | (rj[:, :, None, 1] + 1 == rj[:, None, :, 0])
                    )
                    & (ri[:, :, None, 0] <= ri[:, None, :, 1])
                    & (ri[:, :, None, 1] >= ri[:, None, :, 0])
                )
            )
            # & (rz[:, :, None] == rz[:, None, :])
            & (n[None, :, None] < rn[:, None, None])
            & (n[None, None, :] < n[None, :, None])
        )

    def sample_rworld_states(self, N=1000):
        while True:
            ri = (
                torch.randint(self.height - 2, (N, self.nb_rec_max, 2))
                .sort(dim=2)
                .values
            )
            ri[:, :, 1] += 2
            rj = (
                torch.randint(self.width - 2, (N, self.nb_rec_max, 2))
                .sort(dim=2)
                .values
            )
            rj[:, :, 1] += 2
            rn = torch.randint(self.nb_rec_max - 1, (N,)) + 2
            rz = torch.randint(2, (N, self.nb_rec_max))
            rc = torch.randint(self.nb_colors - 1, (N, self.nb_rec_max)) + 1
            n = torch.arange(self.nb_rec_max)
            nb_collisions = (
                (
                    (ri[:, :, None, 0] <= ri[:, None, :, 1])
                    & (ri[:, :, None, 1] >= ri[:, None, :, 0])
                    & (rj[:, :, None, 0] <= rj[:, None, :, 1])
                    & (rj[:, :, None, 1] >= rj[:, None, :, 0])
                    & (rz[:, :, None] == rz[:, None, :])
                    & (n[None, :, None] < rn[:, None, None])
                    & (n[None, None, :] < n[None, :, None])
                )
                .long()
                .flatten(1)
                .sum(dim=1)
            )

            no_collision = nb_collisions == 0

            if no_collision.any():
                print(no_collision.long().sum() / N)
                self.rn = rn[no_collision]
                self.ri = ri[no_collision]
                self.rj = rj[no_collision]
                self.rz = rz[no_collision]
                self.rc = rc[no_collision]

                nb_contact = (
                    self.contact_matrices(rn, ri, rj, rz).long().flatten(1).sum(dim=1)
                )

                self.rcontact = nb_contact > 0
                self.rfree = torch.full((self.rn.size(0),), True)

                break

    def get_recworld_state(self):
        if not self.rfree.any():
            self.sample_rworld_states()
        k = torch.arange(self.rn.size(0))[self.rfree]
        k = k[torch.randint(k.size(0), (1,))].item()
        self.rfree[k] = False
        return self.rn[k], self.ri[k], self.rj[k], self.rz[k], self.rc[k]

    def draw_state(self, X, rn, ri, rj, rz, rc):
        for n in sorted(list(range(rn)), key=lambda n: rz[n].item()):
            X[ri[n, 0] : ri[n, 1] + 1, rj[n, 0] : rj[n, 1] + 1] = rc[n]

    def task_recworld_immobile(self, A, f_A, B, f_B):
        for X, f_X in [(A, f_A), (B, f_B)]:
            rn, ri, rj, rz, rc = self.get_recworld_state()
            self.draw_state(X, rn, ri, rj, rz, rc)
            ri += 1
            self.draw_state(f_X, rn, ri, rj, rz, rc)

    ######################################################################

    # @torch.compile
    def task_replace_color(self, A, f_A, B, f_B):
        nb_rec = 3
        c = torch.randperm(self.nb_colors - 1)[: nb_rec + 1] + 1
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
        c = torch.randperm(self.nb_colors - 1)[:nb_rec] + 1
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
        c = torch.randperm(self.nb_colors - 1)[:nb_rec] + 1
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
        c = torch.randperm(self.nb_colors - 1)[: 2 * nb_rec] + 1
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
        c = torch.randperm(self.nb_colors - 1)[: nb_rec + 1] + 1
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
        c = torch.randperm(self.nb_colors - 1)[: nb_rec + 1] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            r = self.rec_coo(nb_rec, prevent_overlap=True)
            for n in range(nb_rec):
                i1, j1, i2, j2 = r[n]
                X[i1:i2, j1:j2] = c[n]
                f_X[i1:i2, j1:j2] = c[n]
                if n < nb_rec - 1:
                    for k in range(2):
                        f_X[i1 + k, j1] = c[-1]
                        f_X[i1, j1 + k] = c[-1]

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

    def REMOVED_task_count(self, A, f_A, B, f_B):
        while True:
            error = False

            N = 3
            c = torch.zeros(N + 2, dtype=torch.int64)
            c[1:] = torch.randperm(self.nb_colors - 1)[: N + 1] + 1

            for X, f_X in [(A, f_A), (B, f_B)]:
                if not hasattr(self, "cache_count") or len(self.cache_count) == 0:
                    self.cache_count = list(
                        grow_islands(
                            1000,
                            self.height,
                            self.width,
                            nb_seeds=self.height * self.width // 8,
                            nb_iterations=self.height * self.width // 5,
                        )
                    )

                X[...] = self.cache_count.pop()

                # k = (X.max() + 1 + (c.size(0) - 1)).item()
                # V = torch.arange(k) // (c.size(0) - 1)
                # V = (V + torch.rand(V.size())).sort().indices[: X.max() + 1] % (
                # c.size(0) - 1
                # ) + 1

                V = torch.randint(N, (X.max() + 1,)) + 1
                V[0] = 0
                NB = F.one_hot(c[V]).sum(dim=0)
                X[...] = c[V[X]]
                f_X[...] = X

                if F.one_hot(X.flatten()).max(dim=0).values.sum().item() >= 3:
                    m = NB[c[:-1]].max()
                    if (NB[c[:-1]] == m).long().sum() == 1:
                        for e in range(1, N + 1):
                            if NB[c[e]] == m:
                                a = (f_X == c[e]).long()
                                f_X[...] = (1 - a) * f_X + a * c[-1]
                else:
                    error = True
                    break

            if not error:
                break

        assert F.one_hot(A.flatten()).max(dim=0).values.sum() >= 3

    # @torch.compile
    def task_trajectory(self, A, f_A, B, f_B):
        c = torch.randperm(self.nb_colors - 1)[:2] + 1
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
        c = torch.randperm(self.nb_colors - 1)[:3] + 1
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
                        f_X[i, j] = c[1]

                    if l >= self.width:
                        break

                f_X[i, j] = c[1]
                X[i, j] = c[1]

                if l > 3:
                    break

    # @torch.compile
    def task_scale(self, A, f_A, B, f_B):
        c = torch.randperm(self.nb_colors - 1)[:2] + 1

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

            for k in range(2):
                X[i + k, j] = c[1]
                X[i, j + k] = c[1]
                f_X[i + k, j] = c[1]
                f_X[i, j + k] = c[1]

    # @torch.compile
    def task_symbols(self, A, f_A, B, f_B):
        nb_rec = 4
        c = torch.randperm(self.nb_colors - 1)[: nb_rec + 1] + 1
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

            ai, aj = i.float().mean(), j.float().mean()

            q = torch.randint(3, (1,)).item() + 1

            assert i[q] != ai and j[q] != aj

            for Z in [X, f_X]:
                for k in range(0, nb_rec):
                    Z[i[k] : i[k] + delta, j[k] : j[k] + delta] = c[k]
                # Z[i[0] + delta // 2 - 1, j[0] + delta // 2 - 1] = c[0]
                # Z[i[0] + delta // 2 - 1, j[0] + delta // 2 + 1] = c[0]
                # Z[i[0] + delta // 2 + 1, j[0] + delta // 2 - 1] = c[0]
                # Z[i[0] + delta // 2 + 1, j[0] + delta // 2 + 1] = c[0]

            # f_X[i[0] : i[0] + delta, j[0] : j[0] + delta] = c[q]

            f_X[i[0] + delta // 2, j[0] + delta // 2] = c[q]
            # f_X[i[0] : i[0] + delta, j[0] : j[0] + delta] = c[q]

            ii, jj = (
                i[0] + delta // 2 + (i[q] - ai).sign().long(),
                j[0] + delta // 2 + (j[q] - aj).sign().long(),
            )

            X[ii, jj] = c[nb_rec]
            X[i[0] + delta // 2, jj] = c[nb_rec]
            X[ii, j[0] + delta // 2] = c[nb_rec]

            f_X[ii, jj] = c[nb_rec]
            f_X[i[0] + delta // 2, jj] = c[nb_rec]
            f_X[ii, j[0] + delta // 2] = c[nb_rec]

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

                c = torch.randperm(self.nb_colors - 1)[:nb_rec] + 1

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
    def REMOVED_task_distance(self, A, f_A, B, f_B):
        c = torch.randperm(self.nb_colors - 1)[:3] + 1
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
    def TOO_HARD_task_puzzle(self, A, f_A, B, f_B):
        S = 4
        i0, j0 = (self.height - S) // 2, (self.width - S) // 2
        c = torch.randperm(self.nb_colors - 1)[:4] + 1
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

    def TOO_MESSY_task_islands(self, A, f_A, B, f_B):
        c = torch.randperm(self.nb_colors - 1)[:2] + 1
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
            f_X[...] = (A == A[i, j]) * c[1] + ((A > 0) & (A != A[i, j])) * c[0]
            f_X[i, j] = X[i, j]
            X[i, j] = c[1]

    # @torch.compile
    def TOO_HARD_task_stack(self, A, f_A, B, f_B):
        N = 5
        c = torch.randperm(self.nb_colors - 1)[:N] + 1
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

    def TOO_HARD_task_matrices(self, A, f_A, B, f_B):
        N = 6
        c = torch.randperm(self.nb_colors - 1)[:N] + 1

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

    def TOO_HARD_task_compute(self, A, f_A, B, f_B):
        N = 6
        c = torch.randperm(self.nb_colors - 1)[:N] + 1
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

    # @torch.compile
    # [ai1,ai2] [bi1,bi2]
    def task_contact(self, A, f_A, B, f_B):
        def rec_dist(a, b):
            ai1, aj1, ai2, aj2 = a
            bi1, bj1, bi2, bj2 = b
            v = max(ai1 - bi2, bi1 - ai2)
            h = max(aj1 - bj2, bj1 - aj2)
            return min(max(v, 0) + max(h + 1, 0), max(v + 1, 0) + max(h, 0))

        nb_rec = 3
        c = torch.randperm(self.nb_colors - 1)[:nb_rec] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            while True:
                r = self.rec_coo(nb_rec, prevent_overlap=True)
                d = [rec_dist(r[0], r[k]) for k in range(nb_rec)]
                if min(d[1:]) == 0:
                    break

            for n in range(nb_rec):
                i1, j1, i2, j2 = r[n]
                X[i1:i2, j1:j2] = c[n]
                f_X[i1:i2, j1:j2] = c[n]
                if d[n] == 0:
                    f_X[i1, j1:j2] = c[0]
                    f_X[i2 - 1, j1:j2] = c[0]
                    f_X[i1:i2, j1] = c[0]
                    f_X[i1:i2, j2 - 1] = c[0]

    # @torch.compile
    # [ai1,ai2] [bi1,bi2]
    def task_corners(self, A, f_A, B, f_B):
        polarity = torch.randint(2, (1,)).item()
        nb_rec = 3
        c = torch.randperm(self.nb_colors - 1)[:nb_rec] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            r = self.rec_coo(nb_rec, prevent_overlap=True)

            for n in range(nb_rec):
                i1, j1, i2, j2 = r[n]
                for k in range(2):
                    if polarity == 0:
                        X[i1 + k, j1] = c[n]
                        X[i2 - 1 - k, j2 - 1] = c[n]
                        X[i1, j1 + k] = c[n]
                        X[i2 - 1, j2 - 1 - k] = c[n]
                    else:
                        X[i1 + k, j2 - 1] = c[n]
                        X[i2 - 1 - k, j1] = c[n]
                        X[i1, j2 - 1 - k] = c[n]
                        X[i2 - 1, j1 + k] = c[n]
                    f_X[i1:i2, j1:j2] = c[n]

    def compdist(self, X, i, j):
        dd = X.new_full((self.height + 2, self.width + 2), self.height * self.width)
        d = dd[1:-1, 1:-1]
        m = (X > 0).long()
        d[i, j] = 0
        e = d.clone()
        while True:
            e[...] = d
            d[...] = (
                d.min(dd[:-2, 1:-1] + 1)
                .min(dd[2:, 1:-1] + 1)
                .min(dd[1:-1, :-2] + 1)
                .min(dd[1:-1, 2:] + 1)
            )
            d[...] = (1 - m) * d + m * self.height * self.width
            if e.equal(d):
                break

        return d

    # @torch.compile
    def task_path(self, A, f_A, B, f_B):
        nb_rec = 2
        c = torch.randperm(self.nb_colors - 1)[: nb_rec + 2] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            while True:
                X[...] = 0
                f_X[...] = 0

                r = self.rec_coo(nb_rec, prevent_overlap=True)
                for n in range(nb_rec):
                    i1, j1, i2, j2 = r[n]
                    X[i1:i2, j1:j2] = c[n]
                    f_X[i1:i2, j1:j2] = c[n]

                i1, i2 = torch.randint(self.height, (2,))
                j1, j2 = torch.randint(self.width, (2,))
                if (
                    abs(i1 - i2) + abs(j1 - j2) > 2
                    and X[i1, j1] == 0
                    and X[i2, j2] == 0
                ):
                    d2 = self.compdist(X, i2, j2)
                    d = self.compdist(X, i1, j1)

                    if d2[i1, j1] < 2 * self.width:
                        break

            m = ((d + d2) == d[i2, j2]).long()
            f_X[...] = m * c[-1] + (1 - m) * f_X

            X[i1, j1] = c[-2]
            X[i2, j2] = c[-2]
            f_X[i1, j1] = c[-2]
            f_X[i2, j2] = c[-2]

    # @torch.compile
    def task_fill(self, A, f_A, B, f_B):
        nb_rec = 3
        c = torch.randperm(self.nb_colors - 1)[: nb_rec + 1] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            accept_full = torch.rand(1) < 0.5

            while True:
                X[...] = 0
                f_X[...] = 0

                r = self.rec_coo(nb_rec, prevent_overlap=True)
                for n in range(nb_rec):
                    i1, j1, i2, j2 = r[n]
                    X[i1:i2, j1:j2] = c[n]
                    f_X[i1:i2, j1:j2] = c[n]

                while True:
                    i, j = (
                        torch.randint(self.height, (1,)).item(),
                        torch.randint(self.width, (1,)).item(),
                    )
                    if X[i, j] == 0:
                        break

                d = self.compdist(X, i, j)
                m = (d < self.height * self.width).long()
                X[i, j] = c[-1]
                f_X[...] = m * c[-1] + (1 - m) * f_X
                f_X[i, j] = 0

                if accept_full or (d * (X == 0)).max() == self.height * self.width:
                    break

    def TOO_HARD_task_addition(self, A, f_A, B, f_B):
        c = torch.randperm(self.nb_colors - 1)[:4] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            N1 = torch.randint(2 ** (self.width - 1) - 1, (1,)).item()
            N2 = torch.randint(2 ** (self.width - 1) - 1, (1,)).item()
            S = N1 + N2
            for j in range(self.width):
                r1 = (N1 // (2**j)) % 2
                X[0, -j - 1] = c[r1]
                f_X[0, -j - 1] = c[r1]
                r2 = (N2 // (2**j)) % 2
                X[1, -j - 1] = c[r2]
                f_X[1, -j - 1] = c[r2]
                rs = (S // (2**j)) % 2
                f_X[2, -j - 1] = c[2 + rs]

    def task_science_implicit(self, A, f_A, B, f_B):
        nb_rec = 5
        c = torch.randperm(self.nb_colors - 1)[:nb_rec] + 1

        for X, f_X in [(A, f_A), (B, f_B)]:
            while True:
                i1, i2 = torch.randint(self.height, (2,)).sort().values
                if i1 >= 1 and i2 < self.height and i1 + 3 < i2:
                    break

            while True:
                j1, j2 = torch.randint(self.width, (2,)).sort().values
                if j1 >= 1 and j2 < self.width and j1 + 3 < j2:
                    break

            f_X[i1:i2, j1:j2] = c[0]

            # ---------------------

            while True:
                ii1, ii2 = torch.randint(self.height, (2,)).sort().values
                if ii1 >= i1 and ii2 <= i2 and ii1 + 1 < ii2:
                    break
            jj = torch.randint(j1, (1,))
            X[ii1:ii2, jj:j1] = c[1]
            f_X[ii1:ii2, jj:j1] = c[1]

            while True:
                ii1, ii2 = torch.randint(self.height, (2,)).sort().values
                if ii1 >= i1 and ii2 <= i2 and ii1 + 1 < ii2:
                    break
            jj = torch.randint(self.width - j2, (1,)) + j2 + 1
            X[ii1:ii2, j2:jj] = c[2]
            f_X[ii1:ii2, j2:jj] = c[2]

            # ---------------------

            while True:
                jj1, jj2 = torch.randint(self.width, (2,)).sort().values
                if jj1 >= j1 and jj2 <= j2 and jj1 + 1 < jj2:
                    break
            ii = torch.randint(i1, (1,))
            X[ii:i1, jj1:jj2] = c[3]
            f_X[ii:i1, jj1:jj2] = c[3]

            while True:
                jj1, jj2 = torch.randint(self.width, (2,)).sort().values
                if jj1 >= j1 and jj2 <= j2 and jj1 + 1 < jj2:
                    break
            ii = torch.randint(self.height - i2, (1,)) + i2 + 1
            X[i2:ii, jj1:jj2] = c[4]
            f_X[i2:ii, jj1:jj2] = c[4]

    def task_science_dot(self, A, f_A, B, f_B):
        nb_rec = 3
        c = torch.randperm(self.nb_colors - 1)[: nb_rec + 1] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            while True:
                X[...] = 0
                f_X[...] = 0
                r = self.rec_coo(nb_rec, prevent_overlap=True)
                i, j = (
                    torch.randint(self.height, (1,)).item(),
                    torch.randint(self.width, (1,)).item(),
                )
                q = 0
                for n in range(nb_rec):
                    i1, j1, i2, j2 = r[n]
                    X[i1:i2, j1:j2] = c[n]
                    f_X[i1:i2, j1:j2] = c[n]
                    if i >= i1 and i < i2:
                        q += 1
                        f_X[i, j1:j2] = c[-1]
                    if j >= j1 and j < j2:
                        q += 1
                        f_X[i1:i2, j] = c[-1]
                X[i, j] = c[-1]
                f_X[i, j] = c[-1]
                if q >= 2:
                    break

    def collide(self, s, r, rs):
        i, j = r
        for i2, j2 in rs:
            if abs(i - i2) < s and abs(j - j2) < s:
                return True
        return False

    def task_science_tag(self, A, f_A, B, f_B):
        c = torch.randperm(self.nb_colors - 1)[:4] + 1
        for X, f_X in [(A, f_A), (B, f_B)]:
            rs = []
            while len(rs) < 4:
                i, j = (
                    torch.randint(self.height - 3, (1,)).item(),
                    torch.randint(self.width - 3, (1,)).item(),
                )
                if not self.collide(s=3, r=(i, j), rs=rs):
                    rs.append((i, j))

            for k in range(len(rs)):
                i, j = rs[k]
                q = min(k, 2)
                X[i, j : j + 3] = c[q]
                X[i + 2, j : j + 3] = c[q]
                X[i : i + 3, j] = c[q]
                X[i : i + 3, j + 2] = c[q]

                f_X[i, j : j + 3] = c[q]
                f_X[i + 2, j : j + 3] = c[q]
                f_X[i : i + 3, j] = c[q]
                f_X[i : i + 3, j + 2] = c[q]
                if q == 2:
                    f_X[i + 1, j + 1] = c[-1]

    # end_tasks

    ######################################################################

    def create_empty_quizzes(self, nb, struct=("A", "f_A", "B", "f_B")):
        S = self.height * self.width
        quizzes = torch.zeros(nb, 4 * (S + 1), dtype=torch.int64)
        quizzes[:, 0 * (S + 1)] = self.l2tok[struct[0]]
        quizzes[:, 1 * (S + 1)] = self.l2tok[struct[1]]
        quizzes[:, 2 * (S + 1)] = self.l2tok[struct[2]]
        quizzes[:, 3 * (S + 1)] = self.l2tok[struct[3]]

        return quizzes

    def generate_w_quizzes_(self, nb, tasks=None, progress_bar=False):
        S = self.height * self.width

        if tasks is None:
            tasks = self.all_tasks

        quizzes = self.create_empty_quizzes(nb, ("A", "f_A", "B", "f_B"))

        if progress_bar:
            quizzes = tqdm.tqdm(
                quizzes,
                dynamic_ncols=True,
                desc="world quizzes generation",
                total=quizzes.size(0),
            )

        for quiz in quizzes:
            q = quiz.reshape(4, S + 1)[:, 1:].reshape(4, self.height, self.width)
            q[...] = 0
            A, f_A, B, f_B = q
            task = tasks[torch.randint(len(tasks), (1,)).item()]
            task(A, f_A, B, f_B)

        return quizzes

    def save_some_examples(self, result_dir, prefix=""):
        nb, nrow = 128, 4
        for t in self.all_tasks:
            print(t.__name__)
            quizzes = self.generate_w_quizzes_(nb, tasks=[t])
            self.save_quizzes_as_image(
                result_dir, prefix + t.__name__ + ".png", quizzes, nrow=nrow
            )


######################################################################

if __name__ == "__main__":
    import time

    # grids = Grids(max_nb_cached_chunks=5, chunk_size=100, nb_threads=4)

    grids = Grids()

    # nb = 5
    # quizzes = grids.generate_w_quizzes_(nb, tasks=[grids.task_fill])
    # print(quizzes)
    # print(grids.get_structure(quizzes))
    # quizzes = grids.reconfigure(quizzes, struct=("A", "B", "f_A", "f_B"))
    # print("DEBUG2", quizzes)
    # print(grids.get_structure(quizzes))
    # print(quizzes)

    # i = torch.rand(quizzes.size(0)) < 0.5

    # quizzes[i] = grids.reconfigure(quizzes[i], struct=("f_B", "f_A", "B", "A"))

    # j = grids.indices_select(quizzes, struct=("f_B", "f_A", "B", "A"))

    # print(
    # i.equal(j),
    # grids.get_structure(quizzes[j]),
    # grids.get_structure(quizzes[j == False]),
    # )

    #   exit(0)

    # nb = 1000
    # grids = problem.MultiThreadProblem(
    # grids, max_nb_cached_chunks=50, chunk_size=100, nb_threads=1
    # )
    #    time.sleep(10)
    # start_time = time.perf_counter()
    # prompts, answers = grids.generate_w_quizzes(nb)
    # delay = time.perf_counter() - start_time
    # print(f"{prompts.size(0)/delay:02f} seq/s")
    # exit(0)

    # if True:
    nb, nrow = 128, 4
    # nb, nrow = 8, 2

    # for t in grids.all_tasks:

    for t in [grids.task_recworld_immobile]:
        print(t.__name__)
        w_quizzes = grids.generate_w_quizzes_(nb, tasks=[t])
        grids.save_quizzes_as_image(
            "/tmp",
            t.__name__ + ".png",
            w_quizzes,
            comments=[f"{t.__name__} #{k}" for k in range(w_quizzes.size(0))],
        )

    exit(0)

    nb = 1000

    for t in [
        # grids.task_bounce,
        # grids.task_contact,
        # grids.task_corners,
        # grids.task_detect,
        # grids.task_fill,
        # grids.task_frame,
        # grids.task_grow,
        # grids.task_half_fill,
        # grids.task_isometry,
        # grids.task_path,
        # grids.task_replace_color,
        # grids.task_scale,
        grids.task_symbols,
        # grids.task_trajectory,
        # grids.task_translate,
    ]:
        # for t in [grids.task_path]:
        start_time = time.perf_counter()
        w_quizzes = grids.generate_w_quizzes_(nb, tasks=[t])
        delay = time.perf_counter() - start_time
        print(f"{t.__name__} {w_quizzes.size(0)/delay:02f} seq/s")
        grids.save_quizzes_as_image("/tmp", t.__name__ + ".png", w_quizzes[:128])

    exit(0)

    m = torch.randint(2, (prompts.size(0),))
    predicted_prompts = m * (torch.randint(2, (prompts.size(0),)) * 2 - 1)
    predicted_answers = (1 - m) * (torch.randint(2, (prompts.size(0),)) * 2 - 1)

    grids.save_quizzes_as_image(
        "/tmp",
        "test.png",
        prompts[:nb],
        answers[:nb],
        # You can add a bool to put a frame around the predicted parts
        predicted_prompts[:nb],
        predicted_answers[:nb],
    )
