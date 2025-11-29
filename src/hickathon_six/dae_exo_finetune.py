"""Run supervised finetuning using pretrained DAE fold encoders."""

from hickathon_six import dae_exo
from hickathon_six.DAE import PipelineRunner


def build_config():
    cfg = dae_exo.build_config()
    cfg.load_fold_encoders_from_artifacts = True
    cfg.use_pretrained_full_encoder = True
    return cfg


def main():
    cfg = build_config()
    runner = PipelineRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
