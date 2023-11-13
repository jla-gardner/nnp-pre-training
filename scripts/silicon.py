from itertools import product

from scripts.direct_training import direct_training
from scripts.fine_tuning import fine_tuning

N_PRETRAINS = [100, 330, 1_000, 3_300, 10_000]
N_TRAINS = [25, 50, 100, 200, 400, 800, 1_600]


def run():
    for n_pretrain, structure_type in product(N_PRETRAINS, ("sp2", "sp3")):
        pretrain_config = {
            "n_train": n_pretrain,
            "dataset_name": f"C-SYNTH-23M:{structure_type}",
            "labels": "ace",
        }

        direct_training(**pretrain_config)

        # get the pretrained model
        pretrain_results = direct_training.to_dataframe(config=pretrain_config)
        assert len(pretrain_results) == 1
        pretrain_id = pretrain_results.iloc[0].id

        for seed, n_train in product((1, 2, 3, 4, 42), N_TRAINS):
            direct_training(
                n_train=n_train,
                dataset_name="Si-GAP-18:scaled-to-C",
                labels="dft",
                seed=seed,
            )

            fine_tuning(
                pretrain_id=pretrain_id,
                n_finetune=n_train,
                finetune_dataset="Si-GAP-18:scaled-to-C",
                finetune_labels="dft",
                seed=seed,
            )
