"""audiovj package init.

Contains a CUDA library bootstrap that must run before torch is imported, so it
lives here (the package __init__ is imported before any submodule).
"""

import os
import sys


def _ensure_bundled_cuda_libs() -> None:
    """Make torch's bundled CUDA libs win over a system CUDA install.

    On the GCP Deep Learning VM the system ships CUDA 12.9 under
    ``/usr/local/cuda`` and its ``libcublasLt.so.12`` is registered in the
    ldconfig cache. torch's lazy cuBLASLt loader then picks up that 12.9 library
    instead of its own bundled 12.8 one, and large GEMMs / cuDNN RNNs die with
    ``Invalid handle. Cannot load symbol cublasLtCreate``. Small matmuls dodge
    cuBLASLt and work, which makes ``torch.cuda.is_available()`` misleadingly
    return True.

    The dynamic loader only reads ``LD_LIBRARY_PATH`` at process start, so we
    prepend torch's bundled ``nvidia/*/lib`` directories and re-exec once
    (guarded by a sentinel env var). On CPU-only installs the ``nvidia`` package
    is absent and this is a no-op.
    """
    if os.environ.get("AUDIOVJ_CUDA_BOOTSTRAPPED") == "1":
        return
    os.environ["AUDIOVJ_CUDA_BOOTSTRAPPED"] = "1"

    try:
        import glob

        import nvidia  # installed by torch's CUDA wheels
    except ImportError:
        return  # CPU-only / non-CUDA platform (e.g. macOS): nothing to do

    # A namespace package (no __init__.py) has __file__ == None, which is not an
    # ImportError and so escapes the guard above. Use __path__ instead, which is
    # populated for both regular and namespace packages.
    roots = [r for r in getattr(nvidia, "__path__", []) if r]
    if not roots and getattr(nvidia, "__file__", None):
        roots = [os.path.dirname(nvidia.__file__)]
    libdirs = sorted(
        d for root in roots for d in glob.glob(os.path.join(root, "*", "lib"))
    )
    if not libdirs:
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    # Already prepended (e.g. set by the shell) — don't re-exec.
    if existing.split(os.pathsep)[: len(libdirs)] == libdirs:
        return

    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        libdirs + ([existing] if existing else [])
    )

    # Re-exec is fatal to an interactive REPL; in that case just leave the env
    # set (subprocesses inherit it) rather than killing the session.
    if hasattr(sys, "ps1") or sys.flags.interactive:
        return

    # sys.orig_argv (Py 3.10+) preserves the exact interpreter invocation,
    # including ``-c <cmd>`` / ``-m <mod>`` which sys.argv drops.
    argv = list(getattr(sys, "orig_argv", [sys.executable, *sys.argv]))
    os.execv(sys.executable, argv)


_ensure_bundled_cuda_libs()
