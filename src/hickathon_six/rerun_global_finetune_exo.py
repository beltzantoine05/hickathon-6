"""Rerun the EXO pipeline end-to-end on the full dataset only."""

from hickathon_six import dae_exo
from hickathon_six.DAE import PipelineRunner


def build_config():
    cfg = dae_exo.build_config()
    cfg.run_kfold = False
    cfg.train_full_only = True
    cfg.use_pretrained_full_encoder = False
    cfg.run_full_stage = True
    cfg.embedding_prefix = "exo_embedding"
    cfg.encoder_artifact_name = "exo-global-dae-encoder-full"
    cfg.dae_model_artifact_name = "exo-global-dae-model-full"
    cfg.finetuned_encoder_artifact_name = "exo-global-finetuned-encoder-full"
    cfg.finetuned_model_artifact_name = "exo-global-finetuned-model-full"
    return cfg


def main():
    cfg = build_config()
    runner = PipelineRunner(cfg)
    X_train_df = runner.load_features("X_train.csv")
    X_test_df = runner.load_features("X_test.csv")
    y_train = runner.load_targets()

    X_train_df, y_train = runner._filter_rows_with_missing(X_train_df, y_train)

    encoder_state, _, preprocessor = runner.train_dae_full(X_train_df)
    runner.generate_embeddings(
        X_train_df, X_test_df, encoder_state=encoder_state, preprocessor=preprocessor, skip_finetuned=True
    )

    finetuned_state, finetune_preproc = runner.train_regressor_full(X_train_df, y_train, encoder_state)
    runner.generate_embeddings(
        X_train_df,
        X_test_df,
        encoder_state=encoder_state,
        finetuned_state=finetuned_state,
        preprocessor=finetune_preproc,
    )


if __name__ == "__main__":
    main()
