"""scikit-build-core dynamic-metadata provider for the torch runtime requirement.

torchmcubes links against the C++ ABI of the PyTorch it is compiled with, so the
built wheel must not claim to work with any torch version. This provider pins
the runtime requirement to the minor series of the torch found at build time
(e.g. ``torch==2.14.*``). Patch releases keep the ABI, so they stay allowed;
CUDA/CPU local versions such as ``2.14.0+cu126`` also satisfy the pin.
If torch is not importable at build time the build is aborted with an
explanatory error instead of producing unpinned metadata.
"""

from __future__ import annotations

import importlib.metadata


def torch_requirement() -> str:
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        # Fail fast: without torch the build cannot succeed anyway (CMake needs
        # its config files), and emitting an unpinned requirement would produce
        # wrong metadata. Point the user at the documented install procedure.
        raise RuntimeError(
            "torchmcubes must be built against an installed PyTorch, but the "
            "'torch' package was not found in the build environment. Install "
            "torch first and build without isolation, e.g. "
            "`pip install --no-build-isolation .` (see README)."
        ) from None
    major, minor = version.split(".")[:2]
    return f"torch=={major}.{minor}.*"


class Provider:
    @staticmethod
    def dynamic_metadata(settings, project):
        return {"dependencies": [torch_requirement()]}
