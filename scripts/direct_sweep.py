from scripts.direct_training import direct_training

N_TRAINS = [25, 50, 100, 200, 400, 800, 1600, 3200]


def run():
    for seed in (1, 2, 3, 4, 42):
        for n_train in N_TRAINS:
            direct_training(
                n_train=n_train,
                dataset_name="C-GAP-17",
                labels="dft",
                seed=seed,
            )
