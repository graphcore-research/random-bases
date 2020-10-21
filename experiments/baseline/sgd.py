# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from machinable import Experiment, execute

experiment = Experiment()

for d in ["~cifar10", "~fmnist", "~mnist"]:
    for n in ["~fc", "~cnn"]:
        experiment.component("models.image", (n, d, {"epochs": 100})).name("sgd")

experiment.repeat(3)

if __name__ == "__main__":
    execute(experiment)
