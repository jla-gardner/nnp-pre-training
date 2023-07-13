import os
from datetime import datetime
from pathlib import Path

from rich.progress import BarColumn, Progress, TimeElapsedColumn, TimeRemainingColumn

# this file is in the root/src directory
# the project root is therefore one directory up (i.e. the second parent)
# of this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_debug():
    return os.environ.get("DEBUG", None) == "True"


class Timer:
    def __init__(self):
        self.marks = [("start", datetime.now())]

    def mark(self, name):
        self.marks.append((name, datetime.now()))

    def to_dict(self):
        return {
            name: (end - start).total_seconds()
            for (_, start), (name, end) in zip(self.marks, self.marks[1:])
        }


def list_split(l, n):
    """Split a list into n chunks."""
    chunk_length = len(l) // n
    splits = [chunk_length * i for i in range(1, n)]
    splits.append(len(l))
    splits = [0] + splits
    return [l[splits[i] : splits[i + 1]] for i in range(n)]


def get_model_directory_from(model_id: str):
    root_dir = PROJECT_ROOT / "results"
    matches = list(root_dir.glob(f"**/runs/{model_id}"))
    if len(matches) == 0:
        raise ValueError(f"Model {model_id} not found")
    elif len(matches) > 1:
        raise ValueError(f"Model {model_id} found in multiple places")
    else:
        directory = matches[0]
    return directory


def verbose_iterate(sequence, description=""):
    """
    Print the index of the current iteration.

    Show a description, progress bar, number of completed iterations,
    time elapsed, and estimated time remaining.
    """

    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "{task.completed} / {task.total}",
        "• Elapsed:",
        TimeElapsedColumn(),
        "• Remaining:",
        TimeRemainingColumn(),
    )
    with progress:
        yield from progress.track(
            sequence, total=len(sequence), description=description
        )
