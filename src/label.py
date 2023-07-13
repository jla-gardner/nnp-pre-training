from ase import Atoms
from quippy.potential import Potential

from src.util import verbose_iterate

LABELLERS = {}


def register_labeller(func):
    """Register a function as a labeller."""

    LABELLERS[func.__name__] = func
    return func


def verbose_enumerate(iterable):
    """Print the index of the current iteration."""

    N = len(iterable)
    for i, x in enumerate(iterable):
        print(f"{i} / {N}", end="\r")
        yield i, x


@register_labeller
def gap17(structures):
    """Label structures with GAP17."""

    potential = Potential(param_filename="potentials/gap17/gap17.xml")
    return calculate_labels(potential, structures)


@register_labeller
def si_gap_21(structures):
    """Label structures with the Si GAP 21."""

    potential = Potential(param_filename="potentials/si_gap_21/silicon.xml")
    return calculate_labels(potential, structures)


@register_labeller
def si_gap_18(structures):
    """Label structures with the Si GAP 18."""

    potential = Potential(param_filename="potentials/si_gap_18/gap18.xml")
    return calculate_labels(potential, structures)


@register_labeller
def gap20(structures):
    """Label structures with GAP20."""

    potential = Potential(param_filename="potentials/gap20/gap20.xml")
    return calculate_labels(potential, structures)


@register_labeller
def ace(structures):
    """Label structures with ACE."""
    from pyace import PyACECalculator

    potential = PyACECalculator("potentials/c_ace.yace")
    return calculate_labels(potential, structures)


@register_labeller
def dft(structures):
    """Label structures with DFT."""

    # this is a bit of a hack - all files from load-atoms have
    # an energy and force key as calculated by DFT
    # and we only work with C in this work, so we can just
    # hard code the energy of a lone C atom
    energies = []
    forces = []

    e0s = {"C": -148.314002, "Si": -158.54496821}
    e0 = e0s[structures[0].get_chemical_symbols()[0]]

    for atoms in structures:
        energies.append(atoms.info["energy"] - e0 * len(atoms))
        forces.append(atoms.arrays["force"])
    return energies, forces


@register_labeller
def dft_u(structures):
    """Extract the improved DFT labels from the GAP20 potential."""

    energies = []
    forces = []

    e0 = 0

    for atoms in structures:
        energies.append(atoms.info["energy_U"] - e0 * len(atoms))
        forces.append(atoms.arrays["force_U"])
    return energies, forces


@register_labeller
def lcbop(structures):
    """Label structures with LCBOP."""
    from ase.calculators.kim import KIM

    potential = KIM("Sim_LAMMPS_LCBOP_LosFasolino_2003_C__SM_469631949122_000")
    return calculate_labels(potential, structures)


@register_labeller
def tersoff(structures):
    """Label structures with Tersoff."""
    from ase.calculators.kim import KIM

    potential = KIM("Tersoff_LAMMPS_Tersoff_1988_C__MO_579868029681_003")
    return calculate_labels(potential, structures)


@register_labeller
def edip(structures):
    """Label structures with EDIP."""
    from ase.calculators.kim import KIM

    potential = KIM("EDIP__MD_506186535567_002")
    return calculate_labels(potential, structures)


def lone_atom(structures):
    element = structures[0].get_chemical_symbols()[0]
    atom = Atoms(element)
    atom.center(vacuum=20)
    return atom


def calculate_labels(calculator, structures):
    """Get the labels for a structure using `caluculator`."""

    energies = []
    forces = []

    reference_energy = calculator.get_potential_energy(lone_atom(structures))

    for structure in verbose_iterate(structures, description="Calculating labels:"):
        print(structures.index(structure))
        e = calculator.get_potential_energy(structure)
        e = e - reference_energy * len(structure)
        energies.append(e)
        forces.append(calculator.get_forces(structure))

    return energies, forces
