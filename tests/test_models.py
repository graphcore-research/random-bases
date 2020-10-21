# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from machinable import Execution, Experiment


def _execute(experiment):
    if isinstance(experiment, tuple):
        experiment = Experiment().component(
            *experiment, flags={"OUTPUT_REDIRECTION": "DISABLED"}
        )
    e = (
        Execution(experiment, storage="~/storage/random_bases/tmp")
        .set_behavior(raise_exceptions=True)
        .summary()
        .submit()
    )
    assert e.failures == 0
    return e.storage.get_experiment().components.first()


def _execute_assert(experiment, **kwargs):
    c = _execute(experiment)
    r = c.records.last()
    for k, v in kwargs.items():
        assert float(r[k]) > v


def test_sgd_model(ipu):
    if not ipu:
        return
    _execute_assert(
        (
            "models.image",
            (
                "~fc",
                "~mnist",
                {"epochs": 3},
            ),
        ),
        val_acc=80,
    )


def test_rbd_model(ipu):
    if not ipu:
        return
    _execute_assert(("rbd", ("~fc", "~mnist", {"epochs": 3})), val_acc=60)


def test_fpd_model(ipu):
    if not ipu:
        return
    _execute_assert(
        (
            "rbd",
            (
                "~fc",
                "~mnist",
                {
                    "epochs": 3,
                    "reset_coordinates_each_step": False,
                    "reset_base_each_step": False,
                },
            ),
        ),
        val_acc=40,
    )


def test_nes_model(ipu):
    if not ipu:
        return
    _execute_assert(
        (
            "nes",
            (
                "~fc",
                "~mnist",
                {"epochs": 25},
            ),
        ),
        val_acc=10,
    )


def test_rbd_landscape_model(ipu):
    if not ipu:
        return
    _execute_assert(
        (
            "rbd_landscape",
            (
                "~fc",
                "~mnist",
                {"epochs": 3},
            ),
        ),
        val_acc=50,
    )


def test_rbd_nes_hybrid_model(ipu):
    if not ipu:
        return
    _execute_assert(
        (
            "rbd_nes",
            (
                "~fc",
                "~mnist",
                {"epochs": 3},
            ),
        ),
        val_acc=40,
    )


def test_grad_model(ipu):
    if not ipu:
        return
    _execute_assert(
        (
            "grad",
            (
                "~fc",
                "~mnist",
                {"epochs": 3},
            ),
        ),
        val_acc=70,
    )


def test_rbd_dist_model(ipu):
    if not ipu:
        return
    import ray  # pylint: disable=import-outside-toplevel

    ray.init()
    _execute_assert(
        (
            "rbd_dist",
            (
                "~cnn",
                "~fmnist",
                {"epochs": 1},
            ),
        ),
        val_acc=5,
    )
