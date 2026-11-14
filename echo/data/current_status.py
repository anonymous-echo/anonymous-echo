"""Current observation labels from the final five real frames of each segment."""


def current_label(kind, frame, start, end, classes, tail_frames=5):
    if classes != 3 or tail_frames != 5:
        raise ValueError("Expected three classes and final five real frames")
    if kind not in {"success", "recovery", "failure"} or not start <= frame < end:
        raise ValueError("Expected an actual frame inside a supported segment")
    if frame < max(start, end - tail_frames):
        return 0
    if kind in {"success", "recovery"}:
        return 1
    return 2
