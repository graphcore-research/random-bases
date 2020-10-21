# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import ray
from machinable import Experiment, execute

ray.init()

experiment = (
    Experiment()
    .component("rbd_dist", ("~cnn", "~fmnist", {"workers": 2}))
    .name("rbd_distributed")
)

if __name__ == "__main__":
    execute(experiment)
