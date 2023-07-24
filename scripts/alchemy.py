from itertools import product

from scripts.alchemical_train import alchemical_train
from scripts.fine_tuning import fine_tuning

N_PRETRAINS = [100, 330, 1_000, 3_300, 10_000]
N_TRAINS = [25, 50, 100, 200, 400, 800, 1_600]


def run():
    for pretrain_r_max, n_pretrain in product((4, 6), N_PRETRAINS):
        pretrain_config = {
            "n_train": n_pretrain,
            "dataset_name": "C-SYNTH-23M:reasonable",
            "labels": "ace",
            "r_max": pretrain_r_max,
            "transmute_to": "Si",
            "scale_factor": 1.5,
        }

        alchemical_train(**pretrain_config)

        # get the pretrained model
        pretrain_results = alchemical_train.to_dataframe(config=pretrain_config)
        assert len(pretrain_results) == 1
        pretrain_id = pretrain_results.iloc[0].id

        for seed, n_train in product((1, 2, 3, 4, 42), N_TRAINS):
            fine_tuning(
                pretrain_id=pretrain_id,
                n_finetune=n_train,
                finetune_dataset="Si-GAP-18",
                finetune_labels="dft",
                r_max=4,
                seed=seed,
            )
