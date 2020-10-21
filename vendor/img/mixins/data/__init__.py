# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import math
import multiprocessing

from machinable import Mixin

from .loader import load as load_data


class ImageDataMixin(Mixin):
    def load(self, data=None):
        if data is None:
            data = load_data(**self.config.data.dataset.toDict())
        self.dataset, self.dataset_info = data

        Preprocessor = None
        if (
            self.dataset_info.name == "fashion_mnist"
            or self.dataset_info.name == "mnist"
        ):
            from .preprocessing import MnistPreprocessor as Preprocessor
        elif self.dataset_info.name.startswith("cifar"):
            from .preprocessing import CifarPreprocessor as Preprocessor

        for k, v in self.dataset.items():
            # prefetches a batch at a time to smooth out the time taken to load input
            # files for shuffling and processing
            batch_size = (
                self.config.data.batch_size
                if k == "train"
                else self.config.data.evaluation_batch_size
            )
            self.dataset[k] = v.prefetch(buffer_size=batch_size)

        if Preprocessor is not None:
            options = self.config.data.preprocessing.toDict().copy()
            if options["one_hot"] is True:
                options["one_hot"] = self.dataset_info.features["label"].num_classes
            for k, v in self.dataset.items():
                self.dataset[k] = v.map(
                    Preprocessor(split=k, **options),
                    num_parallel_calls=multiprocessing.cpu_count(),
                )

        # apply shuffle
        train_dataset = self.dataset["train"]
        if self.config.data.shuffle is not False:
            train_dataset = train_dataset.shuffle(
                int(self.config.data.shuffle.buffer),
                seed=self.config.data.shuffle.seed,
                reshuffle_each_iteration=self.config.data.shuffle.each_iteration,
            )

        self.data = {
            "train": train_dataset.repeat().batch(
                self.config.data.batch_size,
                drop_remainder=self.config.data.drop_remainder,
            ),
            "validation": self.dataset["validation"]
            .repeat()
            .batch(
                self.config.data.evaluation_batch_size,
                drop_remainder=self.config.data.drop_remainder,
            ),
            "test": self.dataset["test"]
            .repeat()
            .batch(
                self.config.data.evaluation_batch_size,
                drop_remainder=self.config.data.drop_remainder,
            ),
        }

    def steps_per_epoch(self, split="train"):
        batch_size = (
            self.config.data.batch_size
            if split == "train"
            else self.config.data.evaluation_batch_size
        )
        return math.ceil(self.dataset_info.splits[split].num_examples / batch_size)

    def images_per_epoch(self, split="train"):
        return self.__mixin__.steps_per_epoch(split) * self.config.data.batch_size
