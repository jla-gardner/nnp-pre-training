from scripts.direct_training import direct_training

N_TRAINS = [25, 50, 100, 200, 400, 800, 1_600]


def run():
    for seed in (1, 2, 42):
        for n_train in N_TRAINS:
            direct_training(
                n_train=n_train,
                dataset_name="Si-GAP-18",
                labels="dft",
                r_max=4,
                seed=seed,
            )
