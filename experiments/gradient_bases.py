# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from machinable import Experiment, execute

experiment = (
    Experiment()
    .component("grad", {"base_dimensions": 5})
    .name("gradient_bases")
    .repeat(3)
)

if __name__ == "__main__":
    execute(experiment)
