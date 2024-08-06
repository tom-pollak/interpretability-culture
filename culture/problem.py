#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import threading, queue, torch, tqdm


class Problem:
    def __init__(self, max_nb_cached_chunks=None, chunk_size=None, nb_threads=-1):
        if nb_threads > 0:
            self.chunk_size = chunk_size
            self.queue = queue.Queue(maxsize=max_nb_cached_chunks)
            for _ in range(nb_threads):
                threading.Thread(target=self.fill_cache, daemon=True).start()
            self.rest = None
        else:
            self.queue = None

    def nb_cached_quizzes(self):
        if self.queue is None:
            return None
        else:
            return self.queue.qsize() * self.chunk_size

    def fill_cache(self):
        while True:
            quizzes = self.generate_w_quizzes_(self.chunk_size)
            self.queue.put(quizzes.to("cpu"), block=True)

    def generate_w_quizzes(self, nb, progress_bar=True):
        if self.queue is None:
            return self.generate_w_quizzes_(nb)

        if self.rest is not None:
            quizzes = rest
        else:
            quizzes = []

        self.rest = None

        n = sum([q.size(0) for q in quizzes])

        if progress_bar:
            with tqdm.tqdm(
                total=nb,
                dynamic_ncols=True,
                desc="world generation",
            ) as pbar:
                while n < nb:
                    q = self.queue.get(block=True)
                    quizzes.append(q)
                    n += q.size(0)
                    pbar.update(q.size(0))
        else:
            while n < nb:
                q = self.queue.get(block=True)
                quizzes.append(q)
                n += q.size(0)

        quizzes = torch.cat(quizzes, dim=0)
        assert n == quizzes.size(0)

        k = n - nb

        if k > 0:
            rest = quizzes[-k:]
            quizzes = quizzes[:-k]

        return quizzes

    ######################################################################

    def trivial_prompts_and_answers(self, prompts, answers):
        pass

    # The one to implement, returns two tensors nb x D and nb x D'
    def generate_w_quizzes_(self, nb):
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

    ######################################################################
