from scripts.direct_training import direct_training
from scripts.fine_tuning import fine_tuning

N_PRETRAINS = [100, 330, 1000, 3_300, 10_000, 100_000]
N_TRAINS = [25, 50, 100, 200, 400, 800, 1600, 3200]


def run():
    for n_pretrain in N_PRETRAINS:
        pretrain_config = {
            "n_train": n_pretrain,
            "dataset_name": "C-SYNTH-23M:reasonable",
            "labels": "ace",
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
            )
