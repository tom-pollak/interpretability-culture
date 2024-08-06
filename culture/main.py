#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# > A > f(A) > B ; > f(B)
# < f(B) ; < B < f(A) < A

# Written by Francois Fleuret <francois@fleuret.org>

import math, sys, argparse, time, tqdm, os, datetime, warnings

import torch, torchvision
from torch import nn
from torch.nn import functional as F

import ffutils

import mygpt
import sky, grids, quiz_machine

from quiz_machine import one_batch_masked_inplace_autoregression

import threading, subprocess

import torch.multiprocessing as mp

######################################################################

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--log_filename", type=str, default="train.log")

parser.add_argument("--result_dir", type=str, default=None)

parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--resume", action="store_true", default=False)

parser.add_argument("--max_percents_of_test_in_train", type=int, default=-1)

parser.add_argument("--log_command", type=str, default=None)

# ----------------------------------
parser.add_argument("--nb_epochs", type=int, default=10000)

parser.add_argument("--batch_size", type=int, default=None)

parser.add_argument("--physical_batch_size", type=int, default=None)

parser.add_argument("--inference_batch_size", type=int, default=None)

parser.add_argument("--nb_train_samples", type=int, default=None)

parser.add_argument("--nb_test_samples", type=int, default=None)

parser.add_argument("--nb_new_c_quizzes_for_train", type=int, default=None)

parser.add_argument("--nb_new_c_quizzes_for_test", type=int, default=None)

parser.add_argument("--learning_rate", type=float, default=5e-4)

# ----------------------------------
parser.add_argument("--model", type=str, default=None)

parser.add_argument("--dim_model", type=int, default=None)

parser.add_argument("--dim_keys", type=int, default=None)

parser.add_argument("--dim_hidden", type=int, default=None)

parser.add_argument("--nb_heads", type=int, default=None)

parser.add_argument("--nb_blocks", type=int, default=None)

parser.add_argument("--dropout", type=float, default=0.1)

# ----------------------------------
parser.add_argument("--deterministic_synthesis", action="store_true", default=False)

parser.add_argument("--problem", type=str, default="grids")

parser.add_argument("--nb_threads", type=int, default=1)

parser.add_argument("--gpus", type=str, default="all")

# ----------------------------------

parser.add_argument("--nb_gpts", type=int, default=5)

parser.add_argument("--max_fail_to_validate", type=int, default=2)

parser.add_argument("--accuracy_to_make_c_quizzes", type=float, default=0.98)

parser.add_argument("--proba_understands", type=float, default=0.95)

parser.add_argument("--proba_not_understands", type=float, default=0.5)

parser.add_argument("--temperature_hot", type=float, default=1.5)

parser.add_argument("--temperature_cold", type=float, default=1)

parser.add_argument("--prompt_noise", type=float, default=0.0)

parser.add_argument("--nb_averaging_rounds", type=int, default=3)

parser.add_argument("--dirty_debug", action="store_true", default=False)

parser.add_argument("--test", type=str, default=None)

######################################################################

grids_tasks = ", ".join(
    [x.__name__.removeprefix("task_") for x in grids.Grids().all_tasks]
)

parser.add_argument(
    "--grids_world_tasks",
    type=str,
    default=None,
    help="A comma-separated subset of: " + grids_tasks + ", or None for all.",
)

parser.add_argument(
    "--grids_science_tasks",
    type=str,
    default=None,
    help="A comma-separated subset of: " + grids_tasks + ", or None for all.",
)

######################################################################

parser.add_argument("--sky_height", type=int, default=6)

parser.add_argument("--sky_width", type=int, default=8)

parser.add_argument("--sky_nb_birds", type=int, default=3)

parser.add_argument("--sky_nb_iterations", type=int, default=2)

parser.add_argument("--sky_speed", type=int, default=3)

######################################################################

args = parser.parse_args()

if args.result_dir is None:
    args.result_dir = f"results_culture"

assert not args.grids_science_tasks or (
    len(
        set(args.grids_world_tasks.split(","))
        & set(args.grids_science_tasks.split(","))
    )
    == 0
), "World and science tasks have to be disjoint"

######################################################################

default_args = {
    "model": "37M",
    "batch_size": 25,
    "inference_batch_size": 50,
    "nb_train_samples": 100000,
    "nb_test_samples": 10000,
}

for k, v in default_args.items():
    if getattr(args, k) is None:
        setattr(args, k, v)

######################################################################

default_model_args = {
    "17K": {
        "dim_model": 32,
        "dim_keys": 32,
        "dim_hidden": 32,
        "nb_heads": 2,
        "nb_blocks": 2,
    },
    "4M": {
        "dim_model": 256,
        "dim_keys": 32,
        "dim_hidden": 1024,
        "nb_heads": 4,
        "nb_blocks": 6,
    },
    "37M": {
        "dim_model": 512,
        "dim_keys": 64,
        "dim_hidden": 2048,
        "nb_heads": 8,
        "nb_blocks": 12,
    },
    "122M": {
        "dim_model": 768,
        "dim_keys": 64,
        "dim_hidden": 2048,
        "nb_heads": 8,
        "nb_blocks": 24,
    },
    "352M": {
        "dim_model": 1024,
        "dim_keys": 64,
        "dim_hidden": 2048,
        "nb_heads": 8,
        "nb_blocks": 48,
    },
}

if args.model in default_model_args:
    for k, v in default_model_args[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
else:
    raise ValueError(f"Unknown model {args.model}")

######################################################################

if args.resume:
    assert os.path.isdir(args.result_dir)

else:
    try:
        os.mkdir(args.result_dir)
    except FileExistsError:
        print(f"result directory {args.result_dir} already exists")
        exit(1)

log_file = open(os.path.join(args.result_dir, args.log_filename), "a")

if args.seed >= 0:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

######################################################################


def log_string(s):
    t = time.strftime("%Y%m%d-%H:%M:%S ", time.localtime())

    if log_file is not None:
        log_file.write(t + s + "\n")
        log_file.flush()

    print(t + s)
    sys.stdout.flush()


######################################################################
# Create a time-stamped archive of the source code

with open("this_run.sh", "w") as f:
    f.write(f"{' '.join(sys.argv)}\n")

now = time.strftime("%Y%m%d-%H%M%S", time.localtime())

os.system(f"tar zcvf {args.result_dir}/src-{now}.tgz *.py *.sh")

######################################################################

log_string(f"argv {' '.join(sys.argv)}")

for n in vars(args):
    log_string(f"args.{n} {getattr(args, n)}")


######################################################################

if args.gpus == "all":
    gpus_idx = range(torch.cuda.device_count())
else:
    gpus_idx = [int(k) for k in args.gpus.split(",")]

gpus = [torch.device(f"cuda:{n}") for n in gpus_idx]

if torch.cuda.is_available():
    main_device = gpus[0]
else:
    assert len(gpus) == 0
    main_device = torch.device("cpu")

if args.dirty_debug:
    args.nb_train_samples = 2500
    args.nb_test_samples = 100

if args.physical_batch_size is None:
    args.physical_batch_size = args.batch_size
else:
    assert args.batch_size % args.physical_batch_size == 0

assert args.nb_train_samples % args.batch_size == 0
assert args.nb_test_samples % args.batch_size == 0

if args.problem == "sky":
    problem = sky.Sky(
        height=args.sky_height,
        width=args.sky_width,
        nb_birds=args.sky_nb_birds,
        nb_iterations=args.sky_nb_iterations,
        speed=args.sky_speed,
        max_nb_cached_chunks=len(gpus) * args.nb_train_samples // 100,
        chunk_size=100,
        nb_threads=args.nb_threads,
    )

elif args.problem == "grids":
    problem = grids.Grids(
        max_nb_cached_chunks=len(gpus) * args.nb_train_samples // 100,
        chunk_size=100,
        nb_threads=args.nb_threads,
        tasks=args.grids_world_tasks,
    )

    if args.grids_science_tasks is None:
        science_w_quizzes = None
    else:
        science_problem = grids.Grids(
            max_nb_cached_chunks=len(gpus) * args.nb_train_samples // 100,
            chunk_size=100,
            nb_threads=args.nb_threads,
            tasks=args.grids_science_tasks,
        )
        science_w_quizzes = science_problem.generate_w_quizzes(100)

        if not args.resume:
            science_problem.save_some_examples(args.result_dir, "science_")


else:
    raise ValueError

if not args.resume:
    problem.save_some_examples(args.result_dir)

quiz_machine = quiz_machine.QuizMachine(
    problem=problem,
    batch_size=args.inference_batch_size,
    result_dir=args.result_dir,
    prompt_noise=args.prompt_noise,
    logger=log_string,
    device=main_device,
)

######################################################################

log_string(f"main_device {main_device} gpus {[ str(g) for g in gpus]}")

vocabulary_size = quiz_machine.vocabulary_size()

log_string(f"vocabulary_size {vocabulary_size}")

######################################################################


def run_tests(model, quiz_machine, local_device=main_device):
    with torch.autograd.no_grad():
        model.eval().to(local_device)

        nb_test_samples, acc_test_loss = 0, 0.0
        nb_samples_accumulated = 0

        full_input, _ = quiz_machine.data_input(model, split="test")
        src = full_input.split(args.batch_size)

        for input in tqdm.tqdm(src, dynamic_ncols=True, desc="test"):
            input = input.to(local_device)
            output = model(mygpt.BracketedSequence(input)).x
            loss = F.cross_entropy(output.transpose(1, 2), input)
            acc_test_loss += loss.item() * input.size(0)
            nb_test_samples += input.size(0)

        test_perplexity = math.exp(min(100, acc_test_loss / nb_test_samples))

        log_string(f"test_perplexity {n_epoch} model {model.id} {test_perplexity}")

        model.main_test_accuracy = quiz_machine.produce_results(
            n_epoch=n_epoch,
            model=model,
            input=full_input[:2000],
            result_dir=args.result_dir,
        )


######################################################################


def one_epoch(model, quiz_machine, local_device=main_device):
    model.to(local_device).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    nb_train_samples, acc_train_loss = 0, 0.0

    hard_w_quizzes = []

    full_input, full_from_w = quiz_machine.data_input(model, split="train")
    src = zip(full_input.split(args.batch_size), full_from_w.split(args.batch_size))

    for input, from_w in tqdm.tqdm(
        src,
        dynamic_ncols=True,
        desc="training",
        total=full_input.size(0) // args.batch_size,
    ):
        input = input.to(local_device)

        if nb_train_samples % args.batch_size == 0:
            optimizer.zero_grad()

        targets = input

        output = model(mygpt.BracketedSequence(input)).x
        loss_per_token = F.cross_entropy(
            output.transpose(1, 2), targets, reduction="none"
        )
        loss = loss_per_token.mean()
        acc_train_loss += loss.item() * input.size(0)

        loss_per_samples = loss_per_token.detach().flatten(1).mean(dim=1)
        if from_w.any():
            hard_w_quizzes.append(
                (input[from_w].to("cpu"), loss_per_samples[from_w].to("cpu"))
            )

        nb_train_samples += input.size(0)

        loss.backward()

        if nb_train_samples % args.batch_size == 0:
            optimizer.step()

    train_perplexity = math.exp(min(100, acc_train_loss / nb_train_samples))

    log_string(f"train_perplexity {n_epoch} model {model.id} {train_perplexity}")

    run_tests(model, quiz_machine)

    # threshold = torch.cat([l for _, l in hard_w_quizzes], dim=0).sort().values
    # threshold = threshold[threshold.size(0) // 2]

    # model.hard_w_quizzes = torch.cat(
    # [x[l >= threshold] for x, l in hard_w_quizzes], dim=0
    # )

    model.to(main_device)


######################################################################


def model_transformer_hot(model):
    model.temperature = args.temperature_hot
    # model.set_noise_injection(1.0, ("ffw", args.nb_blocks // 2))


def model_transformer_cold(model):
    model.temperature = args.temperature_cold
    # pass


c_quizzes_procedure = [
    (("f_B", "f_A", "A", "B"), (1, 0, 0, 0), model_transformer_hot),
    (("f_B", "f_A", "A", "B"), (0, 1, 1, 1), model_transformer_cold),
    (("A", "f_A", "B", "f_B"), (0, 0, 0, 1), model_transformer_cold),
    # (("f_B", "f_A", "A", "B"), (0, 0, 1, 1), model_transformer_cold),
    # (("A", "f_A", "B", "f_B"), (0, 0, 0, 1), model_transformer_cold),
]

######################################################################


def save_additional_results(model, models, science_w_quizzes):
    # Save generated quizzes with the successive steps

    recorder = []

    c_quizzes = quiz_machine.generate_c_quizzes(
        64,
        model_for_generation=model,
        procedure=c_quizzes_procedure,
        recorder=recorder,
    )

    ##

    probas = 0

    for a in range(args.nb_averaging_rounds):
        # This is nb_quizzes x nb_models

        seq_logproba = quiz_machine.models_logprobas(
            models, c_quizzes, ("A", "f_A", "B", "f_B"), (0, 0, 0, 1), (0, 0, 1, 0)
        ) + quiz_machine.models_logprobas(
            models, c_quizzes, ("f_A", "A", "f_B", "B"), (0, 0, 0, 1), (0, 0, 1, 0)
        )

        probas += seq_logproba.exp()

    probas /= args.nb_averaging_rounds

    comments = []

    for l in seq_logproba:
        comments.append("proba " + " ".join([f"{x.exp().item():.02f}" for x in l]))

    ##

    c_quizzes = torch.cat([c[:, None, :] for c, _, in recorder], dim=1)
    predicted_parts = torch.cat([t[:, None, :] for _, t in recorder], dim=1)
    nb_steps = c_quizzes.size(1)
    c_quizzes = c_quizzes.reshape(-1, c_quizzes.size(-1))
    predicted_parts = predicted_parts.reshape(-1, predicted_parts.size(-1))

    # We have comments only for the final quiz, not the successive
    # steps, so we have to add nb_steps-1 empty comments

    steps_comments = []
    for c in comments:
        steps_comments += [""] * (nb_steps - 1) + [c]

    filename = f"non_validated_{n_epoch:04d}_{model.id:02d}.png"

    quiz_machine.problem.save_quizzes_as_image(
        args.result_dir,
        filename,
        quizzes=c_quizzes,
        predicted_parts=predicted_parts,
        comments=steps_comments,
        nrow=nb_steps * 2,  # two quiz per row
    )

    log_string(f"wrote {filename}")

    ######################################################################

    if science_w_quizzes is not None:
        struct = ("A", "f_A", "B", "f_B")
        mask = (0, 0, 0, 1)
        result, correct = quiz_machine.predict(
            model=model,
            quizzes=science_w_quizzes.to(main_device),
            struct=struct,
            mask=mask,
        )

        predicted_parts = torch.tensor(mask, device=correct.device)[None, :].expand(
            correct.size(0), -1
        )
        correct = (2 * correct - 1) * (predicted_parts.sum(dim=-1) == 1).long()

        nb_correct = (correct == 1).long().sum()
        nb_total = (correct != 0).long().sum()

        log_string(
            f"science_accuracy {n_epoch} model {model.id} val {nb_correct} / {nb_total}"
        )

        i = correct == 1
        j = correct != 1

        result = torch.cat([result[i], result[j]], dim=0)
        correct = torch.cat([correct[i], correct[j]], dim=0)
        correct_parts = predicted_parts * correct[:, None]

        result = result[:128]
        predicted_parts = predicted_parts[:128]
        correct_parts = correct_parts[:128]

        quiz_machine.problem.save_quizzes_as_image(
            args.result_dir,
            f"culture_science_{n_epoch:04d}_{model.id:02d}.png",
            quizzes=result,
            predicted_parts=predicted_parts,
            correct_parts=correct_parts,
        )


######################################################################


def record_new_c_quizzes(models, quiz_machine, nb_for_train=1000, nb_for_test=100):
    nb_to_validate = nb_for_train + nb_for_test
    nb_to_generate_per_iteration = max(args.physical_batch_size, nb_to_validate)
    nb_validated = 0

    recorded_validated = []

    start_time = time.perf_counter()

    nb_validated_per_model = torch.zeros(len(models), dtype=torch.int64)

    to_recycle = None

    while nb_validated_per_model.sum() < nb_to_validate:
        # We use the model that has generated the fewest quizzes to
        # balance the number of quizzes per model overall

        # model_for_generation = sorted(
        # models, key=lambda m: nb_validated_per_model[m.id]
        # )[0]

        model_for_generation = models[torch.randint(len(models), (1,)).item()]

        # We generate quizzes with a procedure that injects some
        # structured noise

        c_quizzes = quiz_machine.generate_c_quizzes(
            nb_to_generate_per_iteration,
            model_for_generation=model,
            procedure=c_quizzes_procedure,
            to_recycle=to_recycle,
        )

        # We discard the trivial ones, according to a criterion
        # specific to the world quizzes (e.g. B=f(B))

        rejected = []

        to_keep = quiz_machine.problem.trivial(c_quizzes) == False

        if not to_keep.all():
            rejected.append(c_quizzes[to_keep == False])

        c_quizzes = c_quizzes[to_keep]

        probas = 0

        for a in range(args.nb_averaging_rounds):
            # This is nb_quizzes x nb_models

            seq_logproba = quiz_machine.models_logprobas(
                models, c_quizzes, ("A", "f_A", "B", "f_B"), (0, 0, 0, 1), (0, 0, 1, 0)
            ) + quiz_machine.models_logprobas(
                models, c_quizzes, ("f_A", "A", "f_B", "B"), (0, 0, 0, 1), (0, 0, 1, 0)
            )

            probas += seq_logproba.exp()

        probas /= args.nb_averaging_rounds

        nb_succeed = (probas >= args.proba_understands).long().sum(dim=1)
        nb_fail = (probas <= args.proba_not_understands).long().sum(dim=1)

        to_keep = (
            (nb_succeed + nb_fail == probas.size(1))
            & (nb_fail >= 1)
            & (nb_fail <= args.max_fail_to_validate)
        )

        to_recycle = c_quizzes[to_keep == False]
        c_quizzes = c_quizzes[to_keep]

        if c_quizzes.size(0) > 0:
            nb_validated_per_model[model_for_generation.id] += c_quizzes.size(0)
            recorded_validated.append(c_quizzes)
            nb_validated = c_quizzes.size(0)
        else:
            nb_validated = 0

        total_nb_validated = nb_validated_per_model.sum().item()

        duration = time.perf_counter() - start_time

        if total_nb_validated > 0:
            if total_nb_validated < nb_to_validate:
                d = (
                    (nb_to_validate - total_nb_validated)
                    * duration
                    / total_nb_validated
                )
                e = (datetime.datetime.now() + datetime.timedelta(seconds=d)).strftime(
                    "%a %H:%M"
                )
            else:
                e = "now!"
        else:
            e = "???"

        log_string(
            f"keep c_quizzes model {model_for_generation.id} validated {nb_validated} / {nb_to_generate_per_iteration} ({100*nb_validated/nb_to_generate_per_iteration:.02f}%) nb_accumulated {total_nb_validated} / {nb_to_validate} (finishes {e} -- {int((total_nb_validated * 3600)/duration)}/h)"
        )

    validated_quizzes = torch.cat(recorded_validated, dim=0)

    ######################################################################
    # store the new c_quizzes which have been validated

    v_train = validated_quizzes[:nb_for_train]
    quiz_machine.store_c_quizzes(v_train, for_train=True)

    v_test = validated_quizzes[nb_for_train:nb_to_validate]
    quiz_machine.store_c_quizzes(v_test, for_train=False)

    ######################################################################
    # save images

    vq = validated_quizzes[torch.randperm(validated_quizzes.size(0))[:128]]

    if vq.size(0) > 0:
        probas = 0

        for a in range(args.nb_averaging_rounds):
            # This is nb_quizzes x nb_models

            seq_logproba = quiz_machine.models_logprobas(
                models, vq, ("A", "f_A", "B", "f_B"), (0, 0, 0, 1), (0, 0, 1, 0)
            ) + quiz_machine.models_logprobas(
                models, vq, ("f_A", "A", "f_B", "B"), (0, 0, 0, 1), (0, 0, 1, 0)
            )

            probas += seq_logproba.exp()

        probas /= args.nb_averaging_rounds

        comments = []

        for l in seq_logproba:
            comments.append("proba " + " ".join([f"{x.exp().item():.02f}" for x in l]))

        filename = f"culture_c_quiz_{n_epoch:04d}.png"
        quiz_machine.problem.save_quizzes_as_image(
            args.result_dir, filename, vq, comments=comments
        )


######################################################################

# The generator is very similar to a "solving GPT" except that it
# deals with quizzes prologued with one token per solving GPT that
# indicates if the said model solves it or not.
#
# There are three levels of solving 0->proba<=proba_not_understands,
# 2->proba>=proba_understands and 1 otherwise.


def generate_c_quizzes_with_generator(generator, quiz_machine, nb):
    generator.to(main_device)

    struct = ("A", "f_A", "B", "f_B")

    c_quizzes = quiz_machine.problem.create_empty_quizzes(nb, struct=struct)
    ar_mask = quiz_machine.make_ar_mask(c_quizzes, struct, (1, 1, 1, 1))

    i = F.one_hot(
        torch.randint(args.nb_gpts, (c_quizzes.size(0),)),
        num_classes=args.nb_gpts,
    )

    prologs_c_quizzes = token_prolog_0 * i + token_prolog_2 * (1 - i)
    prologs_ar_mask = ar_mask.new_zeros(ar_mask.size(0), prologs_c_quizzes.size(1))

    prologued_c_quizzes = torch.cat([prologs_c_quizzes, c_quizzes], dim=1).to(
        main_device
    )
    prologued_ar_mask = torch.cat([prologs_ar_mask, ar_mask], dim=1).to(main_device)

    seq_logproba = torch.zeros(
        prologued_c_quizzes.size(0), device=prologued_c_quizzes.device
    )

    generator.temperature = args.temperature_hot

    with torch.autograd.no_grad():
        t = generator.training
        generator.eval()

        one_batch_masked_inplace_autoregression(
            generator,
            prologued_c_quizzes,
            prologued_ar_mask,
            seq_logproba,
            deterministic_synthesis=False,
        )

        generator.train(t)

    generator.reset_transformations()

    prologued_c_quizzes = (
        prologued_c_quizzes * (prologued_c_quizzes < vocabulary_size).long()
    )

    c_quizzes = prologued_c_quizzes[:, prologs_c_quizzes.size(1) :]

    return c_quizzes.to("cpu"), prologs_c_quizzes.to("cpu")


def batches_for_generator(generator, quiz_machine, models, fraction_w_quizzes=1.0):
    samples = []

    for _ in range(args.nb_train_samples // args.batch_size):
        while sum([x.size(0) for x in samples]) < args.batch_size:
            # Generate a bunch of quizzes

            if torch.rand(1).item() <= fraction_w_quizzes:
                # Either we start with the world quizzes
                c_quizzes = quiz_machine.problem.generate_w_quizzes(
                    args.batch_size, progress_bar=False
                )
            else:
                # Or we use the generator itself to generate them
                c_quizzes, _ = generate_c_quizzes_with_generator(
                    generator, quiz_machine, args.batch_size
                )

            # We remove the trivial ones
            to_keep = quiz_machine.problem.trivial(c_quizzes) == False
            c_quizzes = c_quizzes[to_keep]

            # If there are remaining ones, we compute the true prolog
            # that indicates how the GPTs solve it

            if c_quizzes.size(0) > 0:
                seq_logproba = quiz_machine.models_logprobas(
                    models,
                    c_quizzes,
                    ("A", "f_A", "B", "f_B"),
                    (0, 0, 0, 1),
                    (0, 0, 1, 0),
                ) + quiz_machine.models_logprobas(
                    models,
                    c_quizzes,
                    ("f_A", "A", "f_B", "B"),
                    (0, 0, 0, 1),
                    (0, 0, 1, 0),
                )

                probas = seq_logproba.exp()

                u0 = probas <= args.proba_not_understands
                u2 = probas >= args.proba_understands
                u1 = (u0 | u2) == False

                prologs = (
                    (u0.long() * token_prolog_0)
                    + (u1.long() * token_prolog_1)
                    + (u2.long() * token_prolog_2)
                )

                prologued_c_quizzes = torch.cat([prologs, c_quizzes], dim=1)

                # nb_u2 = u2.long().sum(dim=1)
                # nb_u0 = u0.long().sum(dim=1)
                # prologued_c_quizzes = prologued_c_quizzes[(nb_u2 >= 1) & (nb_u0 >= 1)]

                if prologued_c_quizzes.size(0) > 0:
                    samples.append(prologued_c_quizzes)

        # Now we yield a batch

        x = torch.cat(samples, dim=0)
        samples = [x[args.batch_size :]]

        yield x[: args.batch_size]


def one_generator_epoch(
    generator, quiz_machine, models, fraction_w_quizzes, local_device=main_device
):
    model.to(local_device).train()

    optimizer = torch.optim.Adam(generator.parameters(), lr=args.learning_rate)

    nb_train_samples, acc_train_loss = 0, 0.0

    src = batches_for_generator(
        generator=generator,
        quiz_machine=quiz_machine,
        models=models,
        fraction_w_quizzes=fraction_w_quizzes,
    )

    for input in tqdm.tqdm(
        src,
        dynamic_ncols=True,
        desc="training",
        total=args.nb_train_samples // args.batch_size,
    ):
        input = input.to(local_device)

        if nb_train_samples % args.batch_size == 0:
            optimizer.zero_grad()

        targets = input

        output = generator(mygpt.BracketedSequence(input)).x
        loss = F.cross_entropy(output.transpose(1, 2), targets)
        acc_train_loss += loss.item() * input.size(0)
        nb_train_samples += input.size(0)

        loss.backward()

        if nb_train_samples % args.batch_size == 0:
            optimizer.step()

    train_perplexity = math.exp(min(100, acc_train_loss / nb_train_samples))

    log_string(f"train_perplexity {n_epoch} generator - {train_perplexity}")

    generator.to(main_device)


######################################################################


def train_complexifier(model_gen, model_pred1, model_pred2):
    samples = []
    perf = []

    optimizer = torch.optim.Adam(model_gen.parameters(), lr=args.learning_rate)

    nb_train_samples, acc_train_loss = 0, 0.0

    for n_epoch in range(args.nb_epochs):
        for b in range(args.nb_train_samples // args.batch_size):
            while sum([x.size(0) for x in samples]) < args.batch_size:
                c_quizzes = quiz_machine.generate_c_quizzes(
                    args.inference_batch_size,
                    model_for_generation=model_gen,
                    procedure=c_quizzes_procedure,
                )
                to_keep = quiz_machine.problem.trivial(c_quizzes) == False
                c_quizzes = c_quizzes[to_keep]
                if c_quizzes.size(0) > 0:
                    seq_logproba = quiz_machine.models_logprobas(
                        [model_pred1, model_pred2],
                        c_quizzes,
                        ("A", "f_A", "B", "f_B"),
                        (0, 0, 0, 1),
                    ) + quiz_machine.models_logprobas(
                        [model_pred1, model_pred2],
                        c_quizzes,
                        ("f_A", "A", "f_B", "B"),
                        (0, 0, 0, 1),
                    )
                    probas = seq_logproba.exp()
                    to_keep = (probas[:, model_pred1.id] >= args.proba_understands) & (
                        probas[:, model_pred2.id] <= args.proba_not_understands
                    )
                    log_string(
                        f"generating {to_keep.long().sum()} / {c_quizzes.size(0)}"
                    )
                    c_quizzes = c_quizzes[to_keep]
                    if c_quizzes.size(0):
                        samples.append(c_quizzes)

            log_string(f"full batch {sum([x.size(0) for x in samples])}")

            x = torch.cat(samples, dim=0)

            input = x[: args.batch_size]
            samples = [x[args.batch_size :]]

            # -------------------

            seq_logproba = quiz_machine.models_logprobas(
                [model_pred1, model_pred2],
                input,
                ("A", "f_A", "B", "f_B"),
                (0, 0, 0, 1),
            ) + quiz_machine.models_logprobas(
                [model_pred1, model_pred2],
                input,
                ("f_A", "A", "f_B", "B"),
                (0, 0, 0, 1),
            )

            comments = []

            for l in seq_logproba:
                comments.append(
                    f"proba {l[model_pred1.id].exp().item():.02f} {l[model_pred2.id].exp().item():.02f}"
                )

            filename = f"batch_{n_epoch:04d}_{b:04d}.png"
            quiz_machine.problem.save_quizzes_as_image(
                args.result_dir, filename, input, comments=comments
            )
            log_string(f"wrote {filename}")

            # ------------------------

            input = input.to(main_device)

            if nb_train_samples % args.batch_size == 0:
                optimizer.zero_grad()

            output = model_gen(mygpt.BracketedSequence(input)).x
            loss = F.cross_entropy(output.transpose(1, 2), input)
            acc_train_loss += loss.item() * input.size(0)
            nb_train_samples += input.size(0)

            loss.backward()

            if nb_train_samples % args.batch_size == 0:
                optimizer.step()

        train_perplexity = math.exp(min(100, acc_train_loss / nb_train_samples))

        log_string(f"train_perplexity {n_epoch} model ae {train_perplexity}")


######################################################################


models = []

for k in range(args.nb_gpts):
    log_string(f"creating model {k} and its w_quizzes")

    model = mygpt.MyGPT(
        vocabulary_size=vocabulary_size,
        dim_model=args.dim_model,
        dim_keys=args.dim_keys,
        dim_hidden=args.dim_hidden,
        nb_heads=args.nb_heads,
        nb_blocks=args.nb_blocks,
        causal=True,
        dropout=args.dropout,
    ).to(main_device)

    model.main_test_accuracy = 0.0
    model.id = k

    model.train_w_quizzes = quiz_machine.problem.generate_w_quizzes(
        args.nb_train_samples
    )

    model.test_w_quizzes = quiz_machine.problem.generate_w_quizzes(args.nb_test_samples)

    models.append(model)

######################################################################

current_epoch = 0

if args.resume:
    for model in models:
        filename = f"gpt_{model.id:03d}.pth"

        try:
            d = torch.load(os.path.join(args.result_dir, filename))
            model.load_state_dict(d[0])
            model.main_test_accuracy = d[1]
            log_string(f"successfully loaded {filename}")
        except FileNotFoundError:
            log_string(f"cannot find {filename}")
            pass

    try:
        filename = "c_quizzes.pth"
        quiz_machine.load_c_quizzes(os.path.join(args.result_dir, filename))
        log_string(f"successfully loaded {filename}")
    except FileNotFoundError:
        log_string(f"cannot find {filename}")
        pass

    try:
        filename = "state.pth"
        state = torch.load(os.path.join(args.result_dir, filename))
        log_string(f"successfully loaded {filename}")
        current_epoch = state["current_epoch"]
    except FileNotFoundError:
        log_string(f"cannot find {filename}")
        pass

######################################################################

nb_parameters = sum(p.numel() for p in models[0].parameters())
log_string(f"nb_parameters {nb_parameters} ({int(nb_parameters/1e6)}M)")

######################################################################

if args.nb_new_c_quizzes_for_train is None:
    args.nb_new_c_quizzes_for_train = args.nb_train_samples // 100

if args.nb_new_c_quizzes_for_test is None:
    args.nb_new_c_quizzes_for_test = args.nb_test_samples // 100

log_string(
    f"nb_new_c_quizzes_for_train {args.nb_new_c_quizzes_for_train} nb_new_c_quizzes_for_test {args.nb_new_c_quizzes_for_test}"
)

######################################################################

if args.dirty_debug:
    args.accuracy_to_make_c_quizzes = 0.0
    args.nb_gpts = 2
    args.nb_new_c_quizzes_for_train = 100
    args.nb_new_c_quizzes_for_test = 10

######################################################################

if args.test == "tsne":
    model = models[0]

    quizzes = []
    labels = []
    nb_samples_per_task = 1000

    for n, t in enumerate(args.grids_world_tasks.split(",")):
        quizzes.append(
            quiz_machine.problem.generate_w_quizzes(nb_samples_per_task, [t])
        )
        labels.append(torch.full((quizzes[-1].size(0),), n))

    quizzes = torch.cat(quizzes, dim=0)
    labels = torch.cat(labels, dim=0)

    with torch.autograd.no_grad():
        model.eval().to(main_device)
        record = []
        for input, targets in zip(
            quizzes.split(args.batch_size), labels.split(args.batch_size)
        ):
            input = input.to(main_device)
            bs = mygpt.BracketedSequence(input)
            bs = mygpt.BracketedSequence(F.pad(bs.x, (1, -1)), bs.first, bs.nb)
            bs = model.embedding(bs)
            bs = model.trunk[args.nb_blocks // 2](bs)
            record.append((bs.x.to("cpu"), targets))

    x = torch.cat([x for x, y in record], dim=0).flatten(1)
    y = torch.cat([y for x, y in record], dim=0)

    print(f"{x.size()=} {y.size()=}")
    # torch.save((x,y), "/tmp/embed.pth")
    # exit(0)

    from sklearn.manifold import TSNE

    x_np = x.numpy()
    z_np = TSNE(n_components=2, perplexity=50).fit_transform(x_np)
    z = torch.from_numpy(z_np)

    print(f"{z.size()=}")

    with open("/tmp/result.dat", "w") as f:
        for k in range(z.size(0)):
            f.write(f"{y[k]} {z[k,0]} {z[k,1]}\n")

    exit(0)

######################################################################

if args.test == "generator":
    token_prolog_0 = vocabulary_size + 0
    token_prolog_1 = vocabulary_size + 1
    token_prolog_2 = vocabulary_size + 2
    generator_vocabulary_size = vocabulary_size + 3

    generator = mygpt.MyGPT(
        vocabulary_size=generator_vocabulary_size,
        dim_model=args.dim_model,
        dim_keys=args.dim_keys,
        dim_hidden=args.dim_hidden,
        nb_heads=args.nb_heads,
        nb_blocks=args.nb_blocks,
        causal=True,
        dropout=args.dropout,
    ).to(main_device)

    generator.main_test_accuracy = 0.0

    filename = f"generator.pth"

    try:
        d = torch.load(os.path.join(args.result_dir, filename))
        generator.load_state_dict(d[0])
        generator.main_test_accuracy = d[1]
        log_string(f"successfully loaded {filename}")
    except FileNotFoundError:
        log_string(f"cannot find {filename}")
        pass

    for n_epoch in range(args.nb_epochs):
        one_generator_epoch(
            generator,
            quiz_machine=quiz_machine,
            models=models,
            fraction_w_quizzes=1 if n_epoch < 25 else 0.5,
            local_device=main_device,
        )

        filename = f"generator.pth"
        torch.save(
            (generator.state_dict(), generator.main_test_accuracy),
            os.path.join(args.result_dir, filename),
        )
        log_string(f"wrote {filename}")

        c_quizzes, prologs = generate_c_quizzes_with_generator(
            generator, quiz_machine, args.batch_size
        )

        seq_logproba = quiz_machine.models_logprobas(
            models, c_quizzes, ("A", "f_A", "B", "f_B"), (0, 0, 0, 1), (0, 0, 1, 0)
        ) + quiz_machine.models_logprobas(
            models, c_quizzes, ("f_A", "A", "f_B", "B"), (0, 0, 0, 1), (0, 0, 1, 0)
        )

        probas = seq_logproba.exp()

        u0 = probas <= args.proba_not_understands
        u2 = probas >= args.proba_understands
        u1 = (u0 | u2) == False

        predicted_prologs = (
            (u0.long() * token_prolog_0)
            + (u1.long() * token_prolog_1)
            + (u2.long() * token_prolog_2)
        )

        comments = []

        nb_errors = (predicted_prologs != prologs).long().sum()
        nb_total = prologs.numel()

        log_string(f"generator_error {nb_errors} / {nb_total}")

        def readable(prologs):
            return (prologs == token_prolog_1) + 2 * (prologs == token_prolog_2)

        for aa, ee, ff in zip(probas, readable(predicted_prologs), readable(prologs)):
            sa = "prolog " + " ".join(
                [f"{e.item()}/{f.item()}" for e, f in zip(ee, ff)]
            )
            sp = "proba " + " ".join([f"{p.item():.02f}" for p in aa])
            comments.append(sa + "\n" + sp)

        filename = f"generator_batch_{n_epoch:04d}.png"
        quiz_machine.problem.save_quizzes_as_image(
            args.result_dir, filename, c_quizzes, comments=comments
        )
        log_string(f"wrote {filename}")

    exit(0)

######################################################################

for n_epoch in range(current_epoch, args.nb_epochs):
    state = {"current_epoch": n_epoch}
    filename = "state.pth"
    torch.save(state, os.path.join(args.result_dir, filename))
    log_string(f"wrote {filename}")

    log_string(f"--- epoch {n_epoch} ----------------------------------------")

    cta = " ".join([f"{float(m.main_test_accuracy):.04f}" for m in models])
    log_string(f"current_test_accuracies {cta}")

    ##################################################
    # If all the models are good enough, generate new quizzes and
    # re-compute the test errors

    if min([m.main_test_accuracy for m in models]) >= args.accuracy_to_make_c_quizzes:
        record_new_c_quizzes(
            models,
            quiz_machine,
            nb_for_train=args.nb_new_c_quizzes_for_train,
            nb_for_test=args.nb_new_c_quizzes_for_test,
        )

        filename = "c_quizzes.pth"
        quiz_machine.save_c_quizzes(os.path.join(args.result_dir, filename))
        log_string(f"wrote {filename}")

        # Force one epoch of training
        for model in models:
            model.main_test_accuracy = 0.0

    ##################################################
    # Select, improve, and eval the worst model(s)

    ranked_models = sorted(models, key=lambda m: float(m.main_test_accuracy))

    weakest_models = ranked_models[: len(gpus)]

    threads = []

    for gpu, model in zip(gpus, weakest_models):
        log_string(f"training model {model.id}")

        t = threading.Thread(
            target=one_epoch, daemon=True, args=(model, quiz_machine, gpu)
        )

        threads.append(t)

        t.start()

    for t in threads:
        t.join()

    # Save the models to disk

    for model in weakest_models:
        filename = f"gpt_{model.id:03d}.pth"
        torch.save(
            (model.state_dict(), model.main_test_accuracy),
            os.path.join(args.result_dir, filename),
        )
        log_string(f"wrote {filename}")

    for model in weakest_models:
        save_additional_results(model, models, science_w_quizzes)

    ######################################################################

    # Renew the training samples

    for model in weakest_models:
        quiz_machine.renew_train_w_quizzes(model=model)

    if args.log_command is not None:
        s = args.log_command.split()
        s.insert(1, args.result_dir)
        subprocess.run(s)

######################################################################
