# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from machinable import Experiment, execute

experiment = Experiment().component(
    "rbd",
    [
        (
            n,
            d,
            {"learning_rate": 2 ** lr, "epochs": 100},
        )
        for d in ["~cifar10", "~fmnist", "~mnist"]
        for n in ["~fc", "~cnn"]
        for lr in [-(q - 5) for q in range(24)]
    ],
)

if __name__ == "__main__":
    execute(experiment)
