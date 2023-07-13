from pathlib import Path
from typing import Union

from digital_experiments import current_directory, experiment, timing
from digital_experiments.util import get_passed_kwargs_for

from src import data, nequip_access
from src.evaluate import evaluate_model
from src.util import PROJECT_ROOT


@experiment(backend="csv", verbose=True, root="../results/fine_tuning", cache=True)
def fine_tuning(
    # pretrained model
    pretrain_id: str,
    checkpoint: Union[str, int] = "best",
    # dataset parameters
    n_finetune=100,
    finetune_dataset="C-GAP-17",
    finetune_labels="dft",
    # model parameters
    r_max=4.0,
    num_layers=4,
    num_features=32,
    l_max=1,
    # training parameters
    seed=42,
    beta=4,
    ema_decay=0.99,
    batch_size=10,
):
    # collect all the kwargs in a dict
    kwargs = locals()
    checkpoint = get_ckpt_file(pretrain_id, checkpoint)

    # fine tune the model
    run_dir = current_directory()
    data.ensure_dataset_exists(finetune_dataset, finetune_labels)

    # run the training
    config = convert_to_direct_training_kwargs(kwargs, "finetune")
    nequip_config = nequip_access.fill_with_defaults(config, run_dir)

    with timing.time_block("training"):
        trainig_results = nequip_access.fine_tune_model(nequip_config, checkpoint)

    # evaluate the model
    with timing.time_block("evaluation"):
        performance_metrics = evaluate_model(
            run_dir, finetune_dataset, finetune_labels, n_finetune
        )

    return {
        **performance_metrics,
        **trainig_results,
    }


def get_ckpt_file(pretrain_id, checkpoint):
    """
    Get the path to the checkpoint file
    """
    best_model = get_best_model_for(pretrain_id)

    if checkpoint == "best":
        return best_model
    else:
        ckpt_file = best_model.parent / f"ckpt{checkpoint}.pth"
        assert ckpt_file.exists(), f"Could not find checkpoint file {ckpt_file}"
        return ckpt_file


def get_best_model_for(experiment_id: str) -> Path:
    """
    find the best_model.pth for the given experiment id
    """
    root = PROJECT_ROOT / "results"
    matches = [*root.glob(f"**/{experiment_id}")]
    assert matches != [], "Could not find any matching files"
    assert len(matches) == 1, "Unexpectedly found more than one matching file"

    return matches[0] / "best_model.pth"


def convert_to_direct_training_kwargs(kwargs, extra_label: str):
    """
    Convert the kwargs for fine tuning to the equivalent ones for direct training
    """
    to_keep = (
        "num_layers r_max num_features seed beta ema_decay l_max batch_size".split()
    )
    to_map = {
        f"n_{extra_label}": "n_train",
        f"{extra_label}_dataset": "dataset_name",
        f"{extra_label}_labels": "labels",
    }
    config = {k: kwargs[k] for k in to_keep}
    config.update({to_map[k]: kwargs[k] for k in to_map})
    return config


def run():
    overrides = get_passed_kwargs_for(fine_tuning)
    fine_tuning(**overrides)
