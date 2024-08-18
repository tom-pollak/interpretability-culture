# %%
from interp.culture import *

import torch as t
import numpy as np
from datasets import DatasetDict, Dataset, Features, ClassLabel, Sequence, Value
from tqdm import tqdm


DATASET_NAME = "tommyp111/culture-puzzles-1M"
NELEMS = 100  # 1_000_000
SEED = 42


def create_grid_dataset() -> Dataset:
    qm = load_quizzes()

    quizzes = qm.problem.all_tasks
    task_names = [t.__name__ for t in quizzes]
    n_per_task = NELEMS // len(quizzes)
    print("Available Tasks:\n- " + "\n- ".join(task_names) + "\n\n---------\n")

    quiz_dict = {"culture": t.cat(qm.test_c_quizzes, dim=0)}
    for task_name, task in tqdm(zip(task_names, quizzes)):
        quiz_dict[task_name.removeprefix("task_")] = qm.problem.generate_w_quizzes_(n_per_task, tasks=[task])  # type: ignore

    features = Features(
        {
            "input_ids": Sequence(Value("int32"), length=405),
            "label": ClassLabel(names=sorted(set(list(quiz_dict.keys())))),
        }
    )

    # concatenate
    quizzes = t.cat(list(quiz_dict.values()), dim=0)

    # randomize configurations
    qm.randomize_configuations_inplace(
        quizzes, structs=[s for s, _, _ in qm.understood_structures]
    )

    # add noise
    if qm.prompt_noise > 0.0:
        for struct, _, noise_mask in qm.understood_structures:
            i = qm.problem.indices_select(quizzes=quizzes, struct=struct)
            if i.any():
                quizzes[i] = qm.problem.inject_noise(
                    quizzes[i], qm.prompt_noise, struct=struct, mask=noise_mask
                )

    # prepend a 0
    quizzes = t.cat(
        (
            t.zeros(quizzes.shape[0], 1, device=quizzes.device, dtype=quizzes.dtype),
            quizzes,
        ),
        dim=1,
    )

    # create labels
    quiz_labels = np.array(
        [
            features["label"].str2int(task_name)
            for task_name, tasks in quiz_dict.items()
            for _ in range(tasks.shape[0])
        ]
    )

    dataset = Dataset.from_dict(
        {"input_ids": quizzes.cpu().numpy(), "label": quiz_labels}, features=features
    )
    dataset.set_format(type="torch", columns=["input_ids", "label"])
    dataset = dataset.shuffle(SEED)
    return dataset


def create_partition_dataset(dataset):
    dataset.set_format(type="torch", columns=["input_ids", "label"])
    labels = dataset.features["label"].names
    dataset_partition = DatasetDict()
    for i, label in enumerate(labels):
        dataset_partition[label] = dataset.filter(
            lambda batch: batch["label"] == i, batched=True
        )
    return dataset_partition


# %%


if __name__ == "__main__":
    print(f"Creating {NELEMS} quizzes...", end="")
    dataset = create_grid_dataset()
    print("done.")

    print("Dataset:")
    print(dataset)
    print("Features:")
    print(dataset.features)
    print("Head:")
    print(dataset[:5])

    dataset_partitioned = create_partition_dataset(dataset)
    print()
    print(dataset_partitioned)
    for task, pd in dataset_partitioned.items():
        print(f"{task}: {len(pd)}")

    print("Pushing to hub...", end="")
    dataset.push_to_hub(DATASET_NAME)
    dataset_partitioned.push_to_hub(DATASET_NAME + "-partitioned")
    print("done.")
