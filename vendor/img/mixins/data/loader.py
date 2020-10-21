# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import random

import tensorflow_datasets as tfds


def load(name, validation_split=False, split_seed=None, data_dir=None, **kwargs):
    if data_dir is None:
        data_dir = "~/tensorflow_datasets"

    if name == "imagenet":
        name = "imagenet2012"

    splits, info = tfds.load(
        name=name,
        data_dir=data_dir,
        split=None,
        with_info=True,
        # don't shuffle files to enable split_seed determinism
        as_dataset_kwargs={"shuffle_files": split_seed is None},
        **kwargs
    )

    if not validation_split:
        # alias
        if "test" in splits and "validation" not in splits:
            splits["validation"] = splits["test"]
            info._splits.add(
                tfds.core.SplitInfo(
                    name="validation",
                    statistics={"num_examples": info.splits["test"].num_examples},
                )
            )
        if "validation" in splits and "test" not in splits:
            splits["test"] = splits["validation"]
            info._splits.add(
                tfds.core.SplitInfo(
                    name="test",
                    statistics={"num_examples": info.splits["validation"].num_examples},
                )
            )

        return splits, info

    # create additional validation split

    if "test" not in splits:
        # use validation as test set
        splits["test"] = splits["validation"]
        info._splits.add(
            tfds.core.SplitInfo(
                name="test",
                statistics={"num_examples": info.splits["validation"].num_examples},
            )
        )

    if validation_split is not True and isinstance(validation_split, int):
        # if int, use absolute number
        num_validation = validation_split
    elif isinstance(validation_split, float):
        # if float, use relative to entire training set
        num_validation = int(
            round(validation_split * info.splits["train"].num_examples)
        )
    else:
        # make it the same size as test data
        num_validation = info.splits["test"].num_examples

    # we just pick the validation split randomly somewhere in the dataset;
    #  that's not really random but good enough for large datasets if they come pre-shuffled and
    #  it saves us from shuffling the entire dataset
    index = random.Random(split_seed).randint(
        0, info.splits["train"].num_examples - num_validation - 1
    )
    splits["validation"] = splits["train"].skip(index).take(num_validation)
    splits["train"] = (
        splits["train"]
        .take(index)
        .concatenate(splits["train"].skip(index + num_validation))
    )
    # for index = 0 that would be equivalent to the deterministic split
    #  splits['validation'] = splits['train'].take(num_validation)
    #  splits['train'] = splits['train'].skip(num_validation)

    # update split info
    try:
        split_info = tfds.core.splits.SplitDict()
    except TypeError:
        # tf-datasets > 1.2
        split_info = tfds.core.splits.SplitDict(dataset_name=name)
    split_info.add(
        tfds.core.SplitInfo(
            name="test", statistics={"num_examples": info.splits["test"].num_examples}
        )
    )
    split_info.add(
        tfds.core.SplitInfo(
            name="validation", statistics={"num_examples": num_validation}
        )
    )
    split_info.add(
        tfds.core.SplitInfo(
            name="train",
            statistics={
                "num_examples": info.splits["train"].num_examples - num_validation
            },
        )
    )
    info._splits = split_info

    return splits, info


def preload(datasets=None, data_dir="~/tensorflow_datasets"):
    """Utility to download multiple datasets at once

    :param datasets: List of datasets; if None all will be downloaded
    """
    if datasets is None:
        datasets = ["mnist", "fashion_mnist", "cifar10", "cifar100"]

    if isinstance(datasets, str):
        datasets = [datasets]

    for dataset in datasets:
        tfds.load(name=dataset, data_dir=data_dir)
