import logging
from pathlib import Path

import ase
import torch
import yaml
from nequip.scripts.deploy import (
    CODE_COMMITS_KEY,
    CONFIG_KEY,
    JIT_BAILOUT_KEY,
    JIT_FUSION_STRATEGY,
    N_SPECIES_KEY,
    R_MAX_KEY,
    TF32_KEY,
    TYPE_NAMES_KEY,
    _compile_for_deploy,
)
from nequip.train import Trainer
from nequip.utils import Config
from nequip.utils._global_options import _set_global_options
from nequip.utils.versions import check_code_version, get_config_code_versions


def deploy(train_dir: Path, model_name: str = "best_model.pth", out_file: Path = None):
    if out_file is None:
        out_file = train_dir / ("deployed_" + model_name)

    if out_file.exists():
        logging.info(f"Model already deployed at {out_file}")
        return out_file

    config = Config.from_file(str(train_dir / "config.yaml"))

    _set_global_options(config)
    check_code_version(config)

    # load the model
    model, _ = Trainer.load_model_from_training_session(
        train_dir, model_name, device="cpu"
    )

    # compile
    model = _compile_for_deploy(model)
    logging.info("Compiled & optimized model.")

    # deploy: verbatim from nequip/nequip/scripts/deploy.py
    metadata: dict = {}
    code_versions, code_commits = get_config_code_versions(config)
    for code, version in code_versions.items():
        metadata[code + "_version"] = version
    if len(code_commits) > 0:
        metadata[CODE_COMMITS_KEY] = ";".join(
            f"{k}={v}" for k, v in code_commits.items()
        )

    metadata[R_MAX_KEY] = str(float(config["r_max"]))
    if "allowed_species" in config:
        # This is from before the atomic number updates
        n_species = len(config["allowed_species"])
        type_names = {
            type: ase.data.chemical_symbols[atomic_num]
            for type, atomic_num in enumerate(config["allowed_species"])
        }
    else:
        # The new atomic number setup
        n_species = str(config["num_types"])
        type_names = config["type_names"]
    metadata[N_SPECIES_KEY] = str(n_species)
    metadata[TYPE_NAMES_KEY] = " ".join(type_names)

    metadata[JIT_BAILOUT_KEY] = str(config[JIT_BAILOUT_KEY])
    if int(torch.__version__.split(".")[1]) >= 11 and JIT_FUSION_STRATEGY in config:
        metadata[JIT_FUSION_STRATEGY] = ";".join(
            "%s,%i" % e for e in config[JIT_FUSION_STRATEGY]
        )
    metadata[TF32_KEY] = str(int(config["allow_tf32"]))
    metadata[CONFIG_KEY] = yaml.dump(dict(config))

    metadata = {k: v.encode("ascii") for k, v in metadata.items()}
    torch.jit.save(model, out_file, _extra_files=metadata)

    return out_file
