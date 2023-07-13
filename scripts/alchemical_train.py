from digital_experiments import current_directory, experiment, timing
from digital_experiments.util import get_passed_kwargs_for

from src import data, nequip_access
from src.evaluate import evaluate_model


@experiment(backend="csv", verbose=True, root="../results/alchemical_train", cache=True)
def alchemical_train(
    # dataset parameters
    n_train=100,
    dataset_name="C-GAP-17",
    labels="dft",
    # transfer parameters
    transmute_to="Si",
    scale_factor=1.5,
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
    config = locals()

    # use the provided current_directory as the root to store training logs etc.
    running_directory = current_directory()

    # ensure the original dataset files exist, and label it if required
    data.ensure_dataset_exists(dataset_name, labels)

    new_dataset_name = data.transmute_dataset(
        dataset_name, labels, transmute_to, scale_factor
    )
    config["dataset_name"] = new_dataset_name

    # run the training
    nequip_config = nequip_access.fill_with_defaults(config, running_directory)

    with timing.time_block("training"):
        training_results = nequip_access.train_model(nequip_config)

    # evaluate the model
    with timing.time_block("evaluation"):
        performance_metrics = evaluate_model(
            running_directory, new_dataset_name, labels, n_train
        )

    return {
        **training_results,
        **performance_metrics,
    }


def run():
    overrides = get_passed_kwargs_for(alchemical_train)
    alchemical_train(**overrides)
