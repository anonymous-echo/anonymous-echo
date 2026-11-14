"""Linux child lifetime guard, installed before loading inference libraries."""

import ctypes
import os
import signal


def bind_to_parent(parent_pid):
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot bind inference lifetime to evaluator")
    if os.getppid() != parent_pid:
        raise RuntimeError("Evaluator exited before inference startup")
