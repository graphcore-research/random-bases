#!/usr/bin/env python3

# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import argparse
import subprocess
import sys


def sh(*cmd):  # pylint: disable=invalid-name
    """Run a shell command, terminating on failure"""
    exitcode = subprocess.call(" ".join(map(str, cmd)), shell=True)
    if exitcode:
        sys.exit(exitcode)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--ci", action="store_true", help="Run in CI mode (check don't autoformat)"
)
args = parser.parse_args()

package = "."  # pylint: disable=invalid-name

# Autoformat

sh(f"black {package}/", "--check" if args.ci else "")

# Lint

sh(f"flake8 {package}/")

sh(f"python -m pylint {package} --ignore {package}/tests")

test_ignores = [
    "missing-module-docstring",
    "missing-function-docstring",
    "missing-class-docstring",
    "redefined-outer-name",
    "unused-argument",
    "protected-access",
    "blacklisted-name",
]
sh(
    f"python -m pylint {package}/tests --ignore test_mixins.py",
    *[f"-d {ignore}" for ignore in test_ignores],
)

# Tests

sh(f"python -m pytest {package}/")
