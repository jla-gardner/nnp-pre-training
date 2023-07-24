from scripts.direct_training import direct_training
from scripts.fine_tuning import fine_tuning

n_pretrain = 10_000
N_TRAINS = [25, 50, 100, 200, 400, 800, 1600, 3200]
pre_train_labels = ["ace", "gap20", "edip", "lcbop"]


def run():
    for seed in (1, 2, 3, 4, 42):
        for label in pre_train_labels:
            pretrain_config = {
                "n_train": n_pretrain,
                "dataset_name": "C-SYNTH-23M:reasonable",
                "labels": label,
            }

            direct_training(**pretrain_config)

            # get the pretrained model
            pretrain_results = direct_training.to_dataframe(config=pretrain_config)
            assert len(pretrain_results) == 1
            pretrain_id = pretrain_results.iloc[0].id

            for n_train in N_TRAINS:
                fine_tuning(
                    pretrain_id=pretrain_id,
                    n_finetune=n_train,
                    finetune_labels="dft",
                    finetune_dataset="C-GAP-17",
                    seed=seed,
                )
