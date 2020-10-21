# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest


@pytest.fixture
def ipu():
    try:
        from tensorflow.python import ipu  # pylint: disable=import-outside-toplevel

        return ipu
    except ImportError:
        return False
