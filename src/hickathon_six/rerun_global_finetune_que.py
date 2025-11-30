"""Rerun QUE finetuning using the pretrained full-dataset DAE encoder."""

from hickathon_six import dae_que
from hickathon_six.DAE import PipelineRunner


def build_config():
    cfg = dae_que.build_config()
    cfg.use_pretrained_full_encoder = True
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
