"""Generate embeddings from pretrained QUE DAE and finetuned encoders."""

from hickathon_six import que_exo
from hickathon_six.DAE import PipelineRunner


def build_config():
    cfg = que_exo.build_config()
    cfg.generate_embeddings_only = True
    cfg.use_pretrained_full_encoder = True
    cfg.wandb.upload_artifacts = False
    return cfg


def main():
    cfg = build_config()
    runner = PipelineRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
