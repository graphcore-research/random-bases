# Improving Neural Network Training in Low Dimensional Random Bases

This repository is the official implementation of Improving Neural Network Training in Low Dimensional Random Bases. 

## Requirements

The implementation requires Python 3 and uses TensorFlow 2. All results in the paper were obtained using Graphcore's version of TensorFlow 2.1 from a pre-release version of the Poplar SDK 1.2; similiar results should be obtained using the Poplar SDK 1.2 release. The distributed implementation uses [Ray](https://github.com/ray-project/ray). The project uses the [machinable](https://machinable.org/) configuration parser. 

We recommend creating a virtual environment before installing the required packages using ``pip install -r requirements.txt``. 

The implementation runs using IPU accelerators or on the CPU.

To adjust any of the default options, edit the [machinable.yaml](./machinable.yaml) configuration file in the root of this repository that contains an overview of all available hyperparameters.

### Using IPU accelerators

Using Graphcore IPUs significantly accelerates training and is enabled by default. 

Please install Poplar SDK 1.2 following the instructions in the Getting Started guide for your IPU system, selecting the TensorFlow 2.1 wheel. See the [Graphcore developer site](https://www.graphcore.ai/developer) for links to the documentation. 

Change the `workers` option to specify how many IPUs to parallelise over (data-parallel training). To split larger models over `N` IPUs in a model-parallel fashion, set `use_sharding: N`.

### Running on the CPU

To run on the CPU, install TensorFlow using `pip install tensorflow==2.1` and set `ipu.enabled: False`.

## Experiments

> Before executing experiments, make sure to change into the root of the repository (e.g. `cd random-bases/`) as the configuration parser uses the `machinable.yaml` in the current working directory by default.

To get started, you can use one of the example scripts in the [experiments](./experiments) directory. For example:
```bash
$ python experiments/baseline/rbd.py
```

In particular, you can use the scripts in the [experiments/baseline](./experiments/baseline) folder to reproduce the results from Table 1.

| Model                  | NES                | FPD                 | RBD                    | SGD               |
|------------------------|--------------------|---------------------|------------------------|-------------------|
| FC-MNIST, D=101,770    | 22.5 +- 1.7 (0.23) | 80 +- 0.4 (0.81)    | 93.893 +- 0.024 (0.96) | 98.27 +- 0.09 (1) |
| FC-FMNIST, D=101,770   | 45 +- 6 (0.52)     | 77.3 +- 0.29 (0.89) | 85.65 +- 0.2 (0.98)    | 87.32 +- 0.21 (1) |
| FC-CIFAR10, D=394,634  | 17.8 +- 0.5 (0.34) | 21.4 +- 1.2 (0.41)  | 43.77 +- 0.22 (0.84)   | 52.09 +- 0.22 (1) |
| CNN-MNIST, D=93,322    | 51 +- 6 (0.51)     | 88.9 +- 0.6 (0.89)  | 97.17 +- 0.1 (0.98)    | 99.41 +- 0.09 (1) |
| CNN-FMNIST, D=93,322   | 37 +- 4 (0.4)      | 77.8 +- 1.6 (0.85)  | 85.56 +- 0.1 (0.93)    | 91.95 +- 0.18 (1) |
| CNN-CIFAR10, D=122,570 | 20.3 +- 1 (0.25)   | 37.2 +- 0.8 (0.46)  | 54.64 +- 0.33 (0.67)   | 81.4 +- 0.4 (1)   |

## Configuring experiments

You can adjust the default parameters in the experiment:
```python
Experiment().component(
    name="rbd",  # name of the model to execute (e.g. rbd, fpd, nes etc.)
    version=(    # hyperparameter changes
        "~cnn",    # adjusts the network and associated learning rate (~cnn or ~fc)
        "~fmnist", # adjust the dataset (~mnist, ~fmnist, ~cifar10)
        {
            "epochs": 100,         # number of epochs
            "base_dimensions": 50  # subspace dimensionality
        }
    )
)
```
You can add multiple models to the same experiment:
```python
# compare evolution strategies model with rbd model for different dimensionality  
Experiment().component("rbd", version={"base_dimensions": 50})\
            .component("nes", version={"base_dimensions": 250})

# compare different dimensions by passing a list of configuration updates
Experiment().component(
    name="rbd",
    version=[
        (
            "~cnn", "~fmnist", { "base_dimensions": d }
        )
        for d in range(50, 550, 50)
    ]
)
```

## Configuring experiment execution

By default, any execution results will be discarded after the execution finishes. Adjust the arguments of the `execute` method to 
```python
from machinable import execute
execute(
    experiment=experiment,  # see above
    storage="~/results",    # where to store results etc.
    engine="native:8",      # execution target, e.g. remote execution, multiprocessing etc.
    seed=123                # random seed
)
```

Learn more about how to execute and adjust the model settings etc. in [machinable's documentation](https://machinable.org/guide/execution.html).

## Adapting the models

The model implementations can be found under [models](./models/), in particular the implementation of the central random bases descent algorithm in [models/rbd.py](./models/rbd.py).
The [vendor](./vendor) directory contains the base implementation of the image classification networks that are extended in this project.
