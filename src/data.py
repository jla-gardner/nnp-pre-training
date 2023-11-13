"""
GitHub poses a limit on file size of 100MB. To circumvent this, we
store the labels in separate files:
    `data/labels/[dataset_name]-[label_type].npz` contains a dict
    of per atom forces ("force") and per cell energies ("energy").
Then we use `load-atoms` to get the actual structures, merge these 
together and store them on disk.

"Raw" structures (i.e. downloaded straight from load-atoms and
with no labels) are in:
    `data/raw/[dataset_name].extxyz`
Labels are in:
    `data/labels/[dataset_name]/[label_type].npz`
"""

import math
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Sequence

import ase
import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from load_atoms import dataset as load_dataset
from locache import persist

from src.label import LABELLERS

DATA_DIR = Path(__file__).parent.parent / "data"
ALL_DATASETS: Dict[str, Callable[[], Sequence[ase.Atoms]]] = {}
"""
A map from dataset name to a function that returns a list of ase.Atoms.
"""


def get_structures_for(dataset_name: str):
    """
    Get the structures for a given dataset.
    """
    if "transmuted" in dataset_name:
        dataset_name = dataset_name.split("transmuted")[0]

    assert dataset_name in ALL_DATASETS, "Unknown dataset"
    load_structures = ALL_DATASETS[dataset_name]
    return load_structures()


def register_dataset(thing=None, *, name=None):
    """
    Register a dataset.

    Can be used to register a dataset:
    - by name (and download this straight from load_atoms)
        >>> register_dataset("C-GAP-17")
    - by function (which should return a list of ase.Atoms)
        >>> @register_dataset(name="C-GAP-20:bucky")
        ... def gap20_bucky():
        ...     pass
    """

    if name is None:
        # called as a function, thing is a load_atoms dataset id
        assert isinstance(thing, str)

        def _loader():
            return load_dataset(thing, root=DATA_DIR / "raw")

        ALL_DATASETS[thing] = _loader

    else:
        # called as a decorator

        # we come through here twice, once with thing=None and once with
        # thing being the function we're decorating

        if thing is None:
            return partial(register_dataset, name=name)

        # thing is the function we're decorating
        assert callable(thing)
        ALL_DATASETS[name] = thing
        return thing


for ds in ["C-GAP-20", "C-SYNTH-1M", "C-SYNTH-23M"]:
    register_dataset(ds)


def has_isolated_atom(structure, cutoff):
    """
    check if any atom in a structure is more than cutoff away from its
    nearest neighbour
    """

    # we do this by building a neighbour list and checking if any atom has
    # no neighbours
    i, j = neighbor_list("ij", structure, cutoff)
    return len(set(i)) != len(structure)


@register_dataset(name="Si-GAP-18")
def si_gap_18():
    dataset = load_dataset("Si-GAP-18", root=DATA_DIR / "raw")
    return [s for s in dataset if not has_isolated_atom(s, 6.0)]


@register_dataset(name="C-GAP-17")
def gap17():
    dataset = load_dataset("C-GAP-17", root=DATA_DIR / "raw")
    return [s for s in dataset if s.info["config_type"] != "dimer"]


@register_dataset(name="C-GAP-20:bucky")
def buckies():
    structures = load_dataset("C-GAP-20", root=DATA_DIR / "raw")
    configs = ["Fullerenes", "Nanotubes"]
    return [s for s in structures if s.info["config_type"] in configs]


@register_dataset(name="C-SYNTH-23M:reasonable")
def reasonable():
    structures = load_dataset("C-SYNTH-23M", root=DATA_DIR / "raw")
    # ignore all structures before 10ps, i.e. very high T
    return [s for s in structures if s.info["time"] >= 10]


@register_dataset(name="C-SYNTH-23M:sp2")
def sp2():
    reasonable_synth = get_structures_for("C-SYNTH-23M:reasonable")
    return [
        structure
        for structure in reasonable_synth
        if 2.0 <= structure.info["density"] < 2.5
    ]


@register_dataset(name="C-SYNTH-23M:sp3")
def sp3():
    reasonable_synth = get_structures_for("C-SYNTH-23M:reasonable")
    return [
        structure
        for structure in reasonable_synth
        if 3.0 < structure.info["density"] <= 3.5
    ]


@register_dataset(name="P-GAP-20:scaled-to-C")
def p_gap_20():
    structures = load_dataset("P-GAP-20", root=DATA_DIR / "raw")
    # scale each of the structures by 1.52
    for s in structures:
        s.cell /= 1.52
        s.positions /= 1.52
    # transmute to C
    for s in structures:
        s.symbols = "C" * len(s)
    # hack energies to be the same as C
    per_atom_energy = np.array(
        [structure.info["energy"] / len(structure) for structure in structures]
    )
    e0 = np.mean(per_atom_energy)
    for s in structures:
        # subtract mean
        s.info["energy"] = s.info["energy"] - e0 * len(s)
        # add mean of carbon so that they have the ~same energy means
        s.info["energy"] += -158 * len(s)
    # filter questionable labels (5 in total) and dimers
    structures = [
        s for s in structures if s.info["energy"] < -8 and len(s) > 2
    ]
    return [s for s in structures if not has_isolated_atom(s, 4 * 1.52)]


@register_dataset(name="Si-GAP-18:scaled-to-C")
def si_gap_scaled():
    structures = load_dataset("Si-GAP-18", root=DATA_DIR / "raw")
    # scale each of the structures by 1.52
    for s in structures:
        s.cell /= 1.61
        s.positions /= 1.61
    # transmute to C
    for s in structures:
        s.symbols = "C" * len(s)
    # hack energies to be the same as C
    per_atom_energy = np.array(
        [structure.info["energy"] / len(structure) for structure in structures]
    )
    e0 = -158.55 + 148.31 - 5
    for s in structures:
        s.info["energy"] = s.info["energy"] - e0 * len(s)

    return [s for s in structures if not has_isolated_atom(s, 4 * 1.61)]


def get_file_path(dataset: str, labels: str, split: str) -> Path:
    """Get the path to a file."""
    filename = f"{split}.extxyz"
    return DATA_DIR / "processed" / dataset / labels / filename


def additional_config(dataset_name: str, labels: str):
    """
    convert a dataset name and label type to a config dict suitable
    for training with nequip.
    """

    train_structures = get_file_path(dataset_name, labels, "train")
    element = read(train_structures, index=0).get_chemical_symbols()[0]

    return dict(
        dataset_file_name=str(train_structures),
        validation_dataset_file_name=str(
            get_file_path(dataset_name, labels, "val")
        ),
        n_val=get_n_val_for(dataset_name),
        key_mapping={
            f"{labels}_energy": "total_energy",
            f"{labels}_force": "forces",
        },
        chemical_symbols=[element],
    )


def ensure_dataset_exists(dataset_name: str, labels: str):
    """Ensure that a dataset exists."""

    train_set_path = get_file_path(dataset_name, labels, "train")
    if not train_set_path.exists():
        create_dataset(dataset_name, labels)


def clean_structures(structures: Sequence[ase.Atoms], to_keep_prefix: str):
    """
    Clean the structures, keeping only the info and arrays with the
    given prefix.
    """

    for s in structures:
        s.calc = None
        s.info = {
            key: value
            for key, value in s.info.items()
            if key.startswith(to_keep_prefix)
        }
        s.arrays = {
            key: value
            for key, value in s.arrays.items()
            if key.startswith(to_keep_prefix)
            or key in ["positions", "numbers"]
        }


def create_dataset(dataset_name: str, labels: str):
    assert (
        dataset_name in ALL_DATASETS
    ), f"Unknown dataset: {dataset_name}. Expected one of {ALL_DATASETS.keys()}"

    print(f"Creating dataset {dataset_name} with labels {labels}")
    # load the structures (and download if required)
    structures = get_structures_for(dataset_name)

    # get the labels
    energies, forces = labels_for_(dataset_name, labels)
    if energies is None:
        energies, forces = generate_labels_for(
            dataset_name, labels, structures
        )

    forces = np.array(forces, dtype=object)
    energies = np.array(energies)
    assert len(structures) == len(energies) == len(forces)

    # add labels to structures
    for i, structure in enumerate(structures):
        structure.info[f"{labels}_energy"] = float(energies[i])
        structure.arrays[f"{labels}_force"] = np.array(
            forces[i], dtype=np.float32
        )

    train, val, test = train_test_split(structures)

    # and clean the structures, so that only the pertinent labels are kept
    clean_structures(train, labels)
    clean_structures(val, labels)
    clean_structures(test, labels)

    # write the files to disk
    save_dataset(dataset_name, labels, train=train, val=val, test=test)


def train_test_split(structures: Sequence[ase.Atoms]):
    if "split" in structures[0].info:
        # split on train/test split
        train = [s for s in structures if s.info["split"] == "train"]
        train = shuffle_structures(train)
        test = [s for s in structures if s.info["split"] == "test"]
    else:
        # random split
        structures = shuffle_structures(structures)
        n_test = min(1_000, len(structures) // 10)
        train, test = structures[:-n_test], structures[-n_test:]

    # split the train set into train and val
    n_val = _get_n_val(train)
    train, val = train[:-n_val], train[-n_val:]

    return train, val, test


def shuffle_structures(structures: Sequence[ase.Atoms]):
    idxs = np.random.RandomState(42).permutation(len(structures))
    return [structures[int(i)] for i in idxs]


def save_dataset(dataset: str, labels, **splits):
    """Write files to disk."""

    for split, split_set in splits.items():
        path = get_file_path(dataset, labels, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, split_set)


def generate_labels_for(
    dataset_name: str, labels: str, structures: Sequence[ase.Atoms]
):
    energies, forces = LABELLERS[labels](structures)

    assert len(energies) == len(forces) == len(structures)

    max_structures_per_file = 10_000

    # split into multiple files
    n_files = math.ceil(len(energies) / max_structures_per_file)
    for i in range(n_files):
        lo = i * max_structures_per_file
        hi = min((i + 1) * max_structures_per_file, len(energies))

        label_path = DATA_DIR / "labels" / dataset_name / f"{labels}-{i}.npz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            label_path,
            energy=energies[lo:hi],
            force=np.array(forces[lo:hi], dtype=object),
        )

    return energies, forces


def labels_for_(dataset_name: str, labels: str):
    """Get the labels for a given dataset."""

    label_dir = DATA_DIR / "labels" / dataset_name
    # label files are of the format <labels>-<i>.npz
    possible_labels = list(label_dir.glob(f"{labels}*.npz"))
    sorted_labels = sorted(
        possible_labels, key=lambda p: int(p.stem.split("-")[-1])
    )

    if len(sorted_labels) == 0:
        return None, None

    energies, forces = [], []

    for label_path in sorted_labels:
        saved_labels = np.load(label_path, allow_pickle=True)
        energies.extend(saved_labels["energy"])
        forces.extend(saved_labels["force"])

    return energies, forces


@persist
def get_n_val_for(dataset_name: str):
    """Get the number of validation structures for a given dataset."""

    structures = get_structures_for(dataset_name)
    _, val, _ = train_test_split(structures)
    return len(val)


def _get_n_val(training_structures: Sequence[ase.Atoms]):
    """Get the number of validation structures for a given training dataset."""

    return min(len(training_structures) // 10, 100)


def transmute_dataset(
    dataset_name: str, labels: str, transmute_to: str, scale_factor: float
):
    new_dataset_name = (
        f"{dataset_name}transmuted:{transmute_to}-{scale_factor}"
    )
    for split in ["train", "val", "test"]:
        new_path = get_file_path(new_dataset_name, labels, split)
        if new_path.exists():
            continue

        old_path = get_file_path(dataset_name, labels, split)
        structures = read(old_path, index=":")
        for s in structures:
            s.symbols = f"{transmute_to}{len(s)}"
            s.cell *= scale_factor
            s.positions *= scale_factor

        new_path.parent.mkdir(parents=True, exist_ok=True)
        write(new_path, structures)

    return new_dataset_name
