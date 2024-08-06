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


class Sky(problem.Problem):
    colors = torch.tensor(
        [
            [255, 255, 255],
            [255, 0, 0],
            [0, 192, 0],
            [0, 0, 255],
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
    first_bird_token = 1
    nb_bird_tokens = colors.size(0) - 1

    token2char = (
        "_" + "".join([chr(ord("A") + n) for n in range(len(colors) - 1)]) + "><"
    )

    def __init__(
        self,
        height=6,
        width=8,
        nb_birds=3,
        speed=2,
        nb_iterations=2,
        avoid_collision=True,
        max_nb_cached_chunks=None,
        chunk_size=None,
        nb_threads=-1,
    ):
        super().__init__(max_nb_cached_chunks, chunk_size, nb_threads)
        self.height = height
        self.width = width
        self.nb_birds = nb_birds
        self.speed = speed
        self.nb_iterations = nb_iterations
        self.avoid_collision = avoid_collision

    def generate_frame_sequences(self, nb):
        frame_sequences = []

        for _ in tqdm.tqdm(range(nb), dynamic_ncols=True, desc="world generation"):
            i, j, vi, vj = (
                torch.empty(self.nb_birds, dtype=torch.int64),
                torch.empty(self.nb_birds, dtype=torch.int64),
                torch.empty(self.nb_birds, dtype=torch.int64),
                torch.empty(self.nb_birds, dtype=torch.int64),
            )

            def collision_okay():
                if not self.avoid_collision:
                    return True

                count = torch.zeros(self.height, self.width, dtype=torch.int64)

                for n in range(self.nb_birds):
                    count[i[n], j[n]] += 1
                    count[i[n] - vi[n], j[n]] += 1
                    count[i[n], j[n] - vj[n]] += 1

                return count.max() <= 1

            col = (
                torch.randperm(self.colors.size(0) - 1)[: self.nb_birds].sort().values
                + 1
            )

            while True:
                while True:
                    for n in range(self.nb_birds):
                        while True:
                            i[n] = torch.randint(self.height, (1,))
                            j[n] = torch.randint(self.width, (1,))
                            vm = torch.randint(4, (1,))
                            vi[n], vj[n] = (vm % 2) * 2 - 1, (vm // 2) * 2 - 1
                            if (
                                i[n] - vi[n] >= 0
                                and i[n] - vi[n] < self.height
                                and j[n] - vj[n] >= 0
                                and j[n] - vj[n] < self.width
                            ):
                                break

                    if collision_okay():
                        break

                result = torch.zeros(
                    self.nb_iterations * self.speed,
                    self.height,
                    self.width,
                    dtype=torch.int64,
                )

                fine = torch.empty(self.nb_iterations * self.speed)

                t_to_keep = (
                    torch.arange(self.nb_iterations, device=result.device) * self.speed
                )

                for l in range(self.nb_iterations * self.speed):
                    fine[l] = collision_okay()
                    for n in range(self.nb_birds):
                        c = col[n]
                        result[l, i[n], j[n]] = c
                        result[l, i[n] - vi[n], j[n]] = c
                        result[l, i[n], j[n] - vj[n]] = c

                        if (i[n] == 0 and vi[n] == -1) or (
                            i[n] == self.height - 1 and vi[n] == 1
                        ):
                            vi[n] = -vi[n]

                        if (j[n] == 0 and vj[n] == -1) or (
                            j[n] == self.width - 1 and vj[n] == 1
                        ):
                            vj[n] = -vj[n]

                        i[n] += vi[n]
                        j[n] += vj[n]

                result = result[t_to_keep]
                fine = fine[t_to_keep]

                if fine[-1]:
                    break

            frame_sequences.append(result)

        return frame_sequences

    ######################################################################

    def frame2img(self, x, scale=15):
        x = x.reshape(x.size(0), self.height, -1)
        m = torch.logical_and(
            x >= 0, x < self.first_bird_token + self.nb_bird_tokens
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

    def seq2str(self, seq):
        result = []
        for s in seq:
            result.append("".join([self.token2char[v] for v in s]))
        return result

    def save_image(
        self,
        result_dir,
        filename,
        prompts,
        answers,
        predicted_prompts=None,
        predicted_answers=None,
    ):
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
                    (c == 1).long() * torch.tensor([0, 255, 0], device=c.device)
                    + (c == 0).long() * torch.tensor([255, 255, 255], device=c.device)
                    + (c == -1).long() * torch.tensor([255, 0, 0], device=c.device)
                )
                y[...] = c[:, :, None, None]

            y[:, :, di : di + x.size(2), dj : dj + x.size(3)] = x

            return y

        margin = 4

        img_prompts = add_frame(self.frame2img(prompts.to("cpu")), c=0, margin=1)
        h = img_prompts.size(2)
        img_answers = add_frame(self.frame2img(answers.to("cpu")), c=0, margin=1)

        img_prompts = add_frame(img_prompts, c=255, margin=margin, bottom=True)
        img_answers = add_frame(img_answers, c=255, margin=margin, bottom=True)

        img_prompts = add_frame(
            img_prompts, c=predicted_prompts, margin=margin, bottom=True
        )
        img_answers = add_frame(
            img_answers, c=predicted_answers, margin=margin, bottom=True
        )

        marker_size = 16

        separator = img_prompts.new_full(
            (
                img_prompts.size(0),
                img_prompts.size(1),
                img_prompts.size(2),
                marker_size,
            ),
            255,
        )

        separator[:, :, 0] = 0
        separator[:, :, h - 1] = 0

        for k in range(1, 2 * marker_size - 8):
            i = k - (marker_size - 4)
            j = marker_size - 5 - abs(i)
            separator[:, :, h // 2 - 1 + i, 2 + j] = 0
            separator[:, :, h // 2 - 1 + i + 1, 2 + j] = 0

        img = torch.cat([img_prompts, separator, img_answers], dim=3)

        image_name = os.path.join(result_dir, filename)
        torchvision.utils.save_image(
            img.float() / 255.0, image_name, nrow=6, padding=margin * 4, pad_value=1.0
        )

    ######################################################################

    def nb_token_values(self):
        return len(self.colors)

    def generate_prompts_and_answers(self, nb):
        frame_sequences = self.generate_frame_sequences(nb)
        frame_sequences = torch.cat([x[None] for x in frame_sequences], dim=0)

        prompts = frame_sequences[:, : frame_sequences.size(1) // 2].flatten(1)

        answers = frame_sequences[:, frame_sequences.size(1) // 2 :].flatten(1)

        # warnings.warn("dirty test with longer answer", RuntimeWarning)
        # answers = torch.cat(
        # [
        # frame_sequences[:, frame_sequences.size(1) // 2 :],
        # frame_sequences[:, frame_sequences.size(1) // 2 :],
        # ],
        # dim=3,
        # ).flatten(1)

        return prompts, answers

    def save_quiz_illustrations(
        self,
        result_dir,
        filename_prefix,
        prompts,
        answers,
        predicted_prompts=None,
        predicted_answers=None,
    ):
        self.save_image(
            result_dir,
            filename_prefix + ".png",
            prompts,
            answers,
            predicted_prompts,
            predicted_answers,
        )


######################################################################

if __name__ == "__main__":
    import time

    sky = Sky(height=6, width=8, speed=1, nb_iterations=4)

    prompts, answers = sky.generate_prompts_and_answers(4)

    predicted_prompts = torch.randint(3, (prompts.size(0),)) - 1
    predicted_answers = torch.randint(3, (prompts.size(0),)) - 1

    sky.save_quiz_illustrations(
        "/tmp", "test", prompts, answers, predicted_prompts, predicted_answers
    )

    # start_time = time.perf_counter()
    # token_sequences = sky.generate_token_sequences(nb=64)
    # delay = time.perf_counter() - start_time
    # print(f"{token_sequences.size(0)/delay:02f} seq/s")

    # print(sky.seq2str(seq[:4]))

    # for t in range(len(it[0])):
    # img = torch.cat([sky.frame2img(f[t]) for f in it], dim=0)
    # torchvision.utils.save_image(
    # img.float() / 255.0,
    # f"/tmp/frame_{t:03d}.png",
    # nrow=8,
    # padding=6,
    # pad_value=0,
    # )

    # m = (torch.rand(seq.size()) < 0.05).long()
    # seq = (1 - m) * seq + m * 23

    # print(seq.size())
    # img = sky.seq2img(token_sequences)
    # print(img.size())

    # torchvision.utils.save_image(
    # img.float() / 255.0, "/tmp/world.png", nrow=6, padding=6, pad_value=0
    # )
