#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import torch
import sys, contextlib

import torch
from torch import Tensor

######################################################################


@contextlib.contextmanager
def evaluation(*models):
    with torch.inference_mode():
        t = [(m, m.training) for m in models]
        for m in models:
            m.train(False)
        yield
        for m, u in t:
            m.train(u)


######################################################################

from torch.utils._python_dispatch import TorchDispatchMode


def hasNaN(x):
    if torch.is_tensor(x):
        return x.isnan().max()
    else:
        try:
            return any([hasNaN(y) for y in x])
        except TypeError:
            return False


class NaNDetect(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        res = func(*args, **kwargs)

        if hasNaN(res):
            raise RuntimeError(
                f"Function {func}(*{args}, **{kwargs}) " "returned a NaN"
            )
        return res


######################################################################


def exception_hook(exc_type, exc_value, tb):
    r"""Hacks the call stack message to show all the local variables
    in case of relevant error, and prints tensors as shape, dtype and
    device.

    """

    repr_orig = Tensor.__repr__
    Tensor.__repr__ = lambda x: f"{x.size()}:{x.dtype}:{x.device}"

    while tb:
        print("--------------------------------------------------\n")
        filename = tb.tb_frame.f_code.co_filename
        name = tb.tb_frame.f_code.co_name
        line_no = tb.tb_lineno
        print(f'  File "{filename}", line {line_no}, in {name}')
        print(open(filename, "r").readlines()[line_no - 1])

        if exc_type in {RuntimeError, ValueError, IndexError, TypeError}:
            for n, v in tb.tb_frame.f_locals.items():
                print(f"  {n} -> {v}")

        print()
        tb = tb.tb_next

    Tensor.__repr__ = repr_orig

    print(f"{exc_type.__name__}: {exc_value}")


def activate_tensorstack():
    sys.excepthook = exception_hook


######################################################################

if __name__ == "__main__":
    import torch

    def dummy(a, b):
        print(a @ b)

    def blah(a, b):
        c = b + b
        dummy(a, c)

    mmm = torch.randn(2, 3)
    xxx = torch.randn(3)
    # print(xxx@mmm)
    blah(mmm, xxx)
    blah(xxx, mmm)
