from itertools import product

from scripts.direct_training import direct_training
from scripts.fine_tuning import fine_tuning

N_TRAINS = [15, 30, 60, 120, 240]


def run():
    general_pretraining = direct_training.to_dataframe(
        config={
            "n_train": 10_000,
            "dataset_name": "C-SYNTH-23M:reasonable",
            "labels": "ace",
        }
    )
    assert len(general_pretraining) == 1
    general_model_id = general_pretraining.iloc[0].id

    for seed, n_train in product((1, 2, 3, 4, 42), N_TRAINS):
        direct_training(
            n_train=n_train,
            dataset_name="C-GAP-20:bucky",
            labels="dft_u",
            seed=seed,
        )

        fine_tuning(
            pretrain_id=general_model_id,
            n_finetune=n_train,
            finetune_dataset="C-GAP-20:bucky",
            finetune_labels="dft_u",
            seed=seed,
        )
