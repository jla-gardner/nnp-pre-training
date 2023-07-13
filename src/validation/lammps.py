import subprocess
import warnings
from pathlib import Path
from typing import Dict, Union

from ase.io import read, write
from digital_experiments import current_directory, experiment

from src.util import PROJECT_ROOT, get_model_directory_from

LAMMPS_DIR = Path(__file__).parent / "lammps"


def run_lammps_script(
    infile: Union[Path, str], variables: Dict, executable: Union[str, Path] = None
):
    """
    Run a LAMMPS script with the given variables.
    """

    if executable is None:
        executable = LAMMPS_DIR / "lmp"
    if isinstance(executable, Path):
        executable = str(executable.resolve())

    if isinstance(infile, Path):
        infile = str(infile.resolve())

    command = [executable, "-in", infile]
    for k, v in variables.items():
        command += ["-var", k, str(v)]

    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            warnings.warn(
                f"LAMMPS simulation failed with return code {process.returncode}.\n{stderr.decode('utf-8')}"
            )
    except:
        pass


@experiment(
    cache=True,
    absolute_root=PROJECT_ROOT / "results/nequip-anneal",
    backend="pickle",
)
def nequip_anneal_run(
    model_id: str,
    starting_density: float = 2.0,
    random_seed: int = 42,
    anneal_T: int = 3000,
    anneal_ps: int = 50,
    warm_ps: int = 20,
    timestep_ps: float = 0.001,
):
    # get deployed model path
    model_dir = get_model_directory_from(model_id).resolve()

    # destination
    destination = current_directory()
    dumps = destination / "dumps"
    dumps.mkdir(parents=True)

    # get structure
    quenched_structures = read(LAMMPS_DIR / "quenched.extxyz", index=":")
    structure = [
        s for s in quenched_structures if s.info["density"] == starting_density
    ][0]
    structure_file = destination / "starting-structure.data"
    write(structure_file, structure, format="lammps-data")

    variables = {
        "structure": structure_file,
        "pair_style": "nequip",
        "pot_file": model_dir / "deployed_best_model.pth",
        "start_T": 300,
        "warm": warm_ps,
        "anneal": anneal_ps,
        "anneal_T": anneal_T,
        "rand_seed": random_seed,
        "dump_dir": dumps.resolve(),
        "logfile": destination.resolve() / "log.lammps",
        "timestep_ps": timestep_ps,
    }

    run_lammps_script(LAMMPS_DIR / "anneal.in", variables)

    # read in dumps: format is dump.<number>.dat
    files = list(dumps.glob("dump*"))
    files.sort(key=lambda x: int(x.name.split(".")[1]))
    structures = [read(f) for f in files]
    for s in structures:
        forces = s.get_forces()
        s.calc = None
        s.arrays["forces"] = forces

    write(destination / "structures.xyz", structures)
    return structures
