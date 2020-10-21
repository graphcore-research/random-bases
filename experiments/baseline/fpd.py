# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from machinable import Experiment, execute

experiment = (
    Experiment()
    .component(
        "rbd",
        (
            "~cnn",
            "~fmnist",
            {
                "epochs": 100,
                "reset_coordinates_each_step": False,
                "reset_base_each_step": False,
            },
        ),
    )
    .name("fpd")
)

if __name__ == "__main__":
    execute(experiment)
