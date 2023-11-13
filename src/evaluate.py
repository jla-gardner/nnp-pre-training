from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Union

import ase
import numpy as np
from ase.io import read
from locache import persist
from nequip.ase import NequIPCalculator

from .data import get_file_path
from .deploy import deploy
from .label import verbose_enumerate
from .util import is_debug, verbose_iterate

EVALUATORS = {}


class Evaluator:
    def __init__(self, calculate: Callable, pre_process: Callable):
        self.collected = []
        self.calculate = calculate
        self.pre_process = pre_process

    def reset(self):
        self.collected = []

    def summarise(self, summarizer):
        return summarizer(self.pre_process(self.collected))

    def __call__(self, *args, **kwargs):
        result = self.calculate(*args, **kwargs)
        self.collected.append(result)
        return result


def register_evaluator(summarizer):
    def decorator(func):
        name = func.__name__
        func = Evaluator(func, summarizer)
        EVALUATORS[name] = func
        return func

    return decorator


@register_evaluator(np.array)
def cell_energy(structure, result, label):
    energy, pred = structure.info[f"{label}_energy"], result["energy"]
    return pred - energy


@register_evaluator(np.array)
def scaled_energy(structure, result, label):
    return cell_energy(structure, result, label) / (len(structure) ** (1 / 2))


@register_evaluator(np.array)
def per_atom_energy(structure, result, label):
    return cell_energy(structure, result, label) / len(structure)


@register_evaluator(np.vstack)
def force(structure, result, label):
    return result["forces"] - structure.arrays[f"{label}_force"]


@persist
def evaluate_model(
    directory: Path,
    dataset: str,
    labels: str,
    n_train: int = None,
    model_name: str = "best_model.pth",
):
    if n_train is None:
        n_train = 10  # default to the first 10 structures
    else:
        n_train = min(
            1_000, n_train
        )  # limit to 1000 structures as this is expensive!

    # first, we deploy the model
    model_path = deploy(directory, model_name)
    model = NequIPCalculator.from_deployed_model(model_path)

    # next we load the structures
    def get_structures(split: str, indices: str):
        return read(get_file_path(dataset, labels, split), index=indices)

    # test on all the val, all the test, and the first n_train of the train
    indices = dict(train=f":{n_train}", val=":", test=":")
    structures_by_split = {
        split: get_structures(split, index) for split, index in indices.items()
    }

    # finally we return the metrics
    return {
        split: evaluate(model, structures, labels)
        for split, structures in structures_by_split.items()
    }


def get_model(
    x: Union[str, Path],
    checkpoint_name: str = "best_model.pth",
) -> NequIPCalculator:
    # if model is a string (i.e. id), we work out where it's folder is
    if isinstance(x, str):
        root_dir = Path(__file__).parent.parent / "results"
        matches = list(root_dir.glob(f"**/runs/{x}"))
        if len(matches) == 0:
            raise ValueError(f"Model {x} not found")
        elif len(matches) > 1:
            raise ValueError(f"Model {x} found in multiple places")
        else:
            directory = matches[0]
    else:
        directory = x

    model_path = deploy(directory, checkpoint_name)
    return NequIPCalculator.from_deployed_model(model_path)


@persist
def get_model_predictions(
    model: Union[str, Path],
    structures: list[ase.Atoms],
    model_name: str = "best_model.pth",
):
    """
    Get the predictions of a model on a set of structures.
    """

    nequip = get_model(model, model_name)

    # then we make predictions
    predictions = []
    print(f"Getting predictions for {len(structures)} structures...")
    for _, structure in verbose_enumerate(structures):
        nequip.calculate(structure, ["energy", "forces"])
        predictions.append(nequip.results)

    return predictions


def evaluate(
    model: NequIPCalculator, structures: list[ase.Atoms], label: str
) -> Dict[str, float]:
    if is_debug():
        structures = structures[:10]

    for evaluator in EVALUATORS.values():
        evaluator.reset()

    print(f"Evaluating model on {len(structures)} structures...")
    for structure in verbose_iterate(structures):
        model.calculate(structure, ["energy", "forces"])

        for evaluator in EVALUATORS.values():
            evaluator(structure, model.results, label)

    rmse = lambda x: np.sqrt(np.mean(x**2))
    return {
        f"{name}_rmse": evaluator.summarise(rmse)
        for name, evaluator in EVALUATORS.items()
    }
