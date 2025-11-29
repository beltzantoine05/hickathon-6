"""Train a single DAE fold (default: fold 0) and save the encoder locally."""

import argparse

from hickathon_six import dae_exo
from hickathon_six.DAE import PipelineRunner


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index to pretrain (default: 0).",
    )
    return parser.parse_args()


def build_config(fold: int):
    cfg = dae_exo.build_config()
    cfg.fold_indices = [fold]
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args.fold)
    runner = PipelineRunner(cfg)
    X_train_df = runner.load_features("X_train.csv")
    X_train_df, _ = runner._filter_rows_with_missing(X_train_df)
    runner.train_dae_kfold(X_train_df)


if __name__ == "__main__":
    main()
