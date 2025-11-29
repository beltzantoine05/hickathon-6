"""Run supervised finetuning using pretrained DAE fold encoders."""

import argparse

from hickathon_six import dae_exo
from hickathon_six.DAE import PipelineRunner


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Optional fold index to finetune. If omitted, all folds are processed.",
    )
    return parser.parse_args()


def build_config(fold: int | None = None):
    cfg = dae_exo.build_config()
    cfg.load_fold_encoders_from_artifacts = True
    cfg.use_pretrained_full_encoder = True
    if fold is not None:
        cfg.fold_indices = [fold]
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args.fold)
    runner = PipelineRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
