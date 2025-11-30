"""Rerun EXO finetuning using the pretrained full-dataset DAE encoder."""

from hickathon_six import dae_exo
from hickathon_six.DAE import PipelineRunner


def build_config():
    cfg = dae_exo.build_config()
    cfg.use_pretrained_full_encoder = True
    cfg.train_full_only = True
    cfg.run_kfold = False
    cfg.run_full_stage = True
    cfg.encoder_artifact_name = "exo-dae-encoder-full"
    cfg.finetuned_encoder_artifact_name = "exo-finetuned-encoder-full"
    cfg.dae_model_artifact_name = "exo-dae-model-full"
    cfg.finetuned_model_artifact_name = "exo-finetuned-model-full"
    return cfg


def main():
    cfg = build_config()
    runner = PipelineRunner(cfg)
    X_train_df = runner.load_features("X_train.csv")
    X_test_df = runner.load_features("X_test.csv")
    X_train_df, _ = runner._filter_rows_with_missing(X_train_df)
    runner.generate_embeddings(X_train_df, X_test_df, skip_finetuned=True)
    runner.run()


if __name__ == "__main__":
    main()
