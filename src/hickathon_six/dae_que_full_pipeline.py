"""Run only the full-dataset QUE DAE pretraining, finetuning, and embedding generation."""

from hickathon_six import dae_que
from hickathon_six.DAE import PipelineRunner


def build_config():
    cfg = que_exo.build_config()
    cfg.train_full_only = True
    cfg.run_kfold = False
    cfg.run_full_stage = True
    return cfg


def main():
    cfg = build_config()
    runner = PipelineRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
