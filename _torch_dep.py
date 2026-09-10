"""scikit-build-core dynamic-metadata provider for the torch runtime requirement.

torchmcubes links against the C++ ABI of the PyTorch it is compiled with, so the
built wheel must not claim to work with any torch version. This provider pins
the runtime requirement to the minor series of the torch found at build time
(e.g. ``torch==2.14.*``). Patch releases keep the ABI, so they stay allowed;
CUDA/CPU local versions such as ``2.14.0+cu126`` also satisfy the pin.
"""

from __future__ import annotations

import importlib.metadata


def torch_requirement() -> str:
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        # No torch at build time: CMake will fail later with a clear message,
        # so just fall back to an unpinned requirement here.
        return "torch"
    major, minor = version.split(".")[:2]
    return f"torch=={major}.{minor}.*"


class Provider:
    @staticmethod
    def dynamic_metadata(settings, project):
        return {"dependencies": [torch_requirement()]}
