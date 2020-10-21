# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from machinable import Experiment, execute

experiment = (
    Experiment().component("rbd", ("~cnn", "~fmnist", {"epochs": 100})).name("rbd")
)

if __name__ == "__main__":
    execute(experiment)
