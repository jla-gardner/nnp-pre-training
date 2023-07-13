from pathlib import Path
from typing import Any, Dict

import yaml
from nequip.scripts._logger import set_up_script_logger
from nequip.scripts.train import default_config as nequip_defaults
from nequip.scripts.train import fresh_start, restart
from nequip.utils import Config, load_file

from . import data, util

_ME = Path(__file__).resolve()

set_up_script_logger(None)


def get_restart_file(config: Config):
    return Path(config.root) / config.run_name / "trainer.pth"


def fill_with_defaults(config: Dict[str, Any], running_directory: Path):
    # add required information
    config["root"] = running_directory.parent
    config["run_name"] = running_directory.name
    config["dataset_seed"] = config["seed"]

    # generate the loss function
    if "beta" in config:
        config["loss_coeffs"] = dict(
            forces=1, total_energy=[config["beta"], "PerAtomMSELoss"]
        )

    # convert the dataset name etc. to valid nequip config
    config.update(data.additional_config(config["dataset_name"], config["labels"]))

    # add the defaults, both from the nequip defaults...
    all_config = Config.from_dict(nequip_defaults)

    # ... and those I've defined in the defaults.yaml file
    with open(_ME.parent / "defaults.yaml", encoding="utf-8") as f:
        my_defaults = yaml.load(f, Loader=yaml.FullLoader)
    all_config.update(my_defaults)

    all_config.update(config)

    # add the debug flag
    if util.is_debug():
        all_config["max_epochs"] = 10

    return all_config


def train_model(config: Config):
    # we train our models in one go:
    # no restart files for this config should exist
    found_restart_file = get_restart_file(config).exists()
    assert not found_restart_file, "Training model: no restart file should exist"
    print(config["dataset_file_name"])
    trainer = fresh_start(config)

    trainer.save()
    trainer.train()

    num_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    return dict(
        num_params=num_params, best_epoch=trainer.best_epoch, num_epochs=trainer.iepoch
    )


def fine_tune_model(config: Config, pretrained_file: Path):
    # we modify the config so that NequIP will load the pretrained model
    # weights from the file
    # see https://github.com/mir-group/nequip/discussions/235
    config["model_builders"].append("initialize_from_state")
    config["initial_model_state"] = pretrained_file

    # dont care about checkpoint files
    config["save_checkpoint_freq"] = -1

    # then training is the same as direct training
    return train_model(config)
