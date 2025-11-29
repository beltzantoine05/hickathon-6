"""Example runner for the configurable DAE pipeline."""
from hickathon_six.DAE import (
    ArchitectureConfig,
    DAETrainingConfig,
    PipelineConfig,
    PreprocessingConfig,
    RegressionTrainingConfig,
    WandbConfig,
    PipelineRunner,
)

COLUMNS_TO_LOAD = [
    # Reading average scores
    'reading_q1_average_score', 'reading_q2_average_score', 'reading_q3_average_score',
    'reading_q4_average_score', 'reading_q5_average_score', 'reading_q6_average_score',
    'reading_q7_average_score', 'reading_q8_average_score', 'reading_q9_average_score',
    'reading_q10_average_score', 'reading_q11_average_score', 'reading_q12_average_score',
    'reading_q13_average_score', 'reading_q14_average_score', 'reading_q15_average_score',
    # Math average scores
    'math_q1_average_score', 'math_q2_average_score', 'math_q3_average_score',
    'math_q4_average_score', 'math_q5_average_score', 'math_q6_average_score',
    'math_q7_average_score', 'math_q8_average_score', 'math_q9_average_score',
    'math_q10_average_score', 'math_q11_average_score', 'math_q12_average_score',
    'math_q13_average_score', 'math_q14_average_score', 'math_q15_average_score',
    'math_q16_average_score', 'math_q17_average_score', 'math_q18_average_score',
    'math_q19_average_score', 'math_q20_average_score', 'math_q21_average_score',
    # Science average scores
    'science_q1_average_score', 'science_q2_average_score', 'science_q3_average_score',
    'science_q4_average_score', 'science_q5_average_score', 'science_q6_average_score',
    'science_q7_average_score', 'science_q8_average_score', 'science_q9_average_score',
    'science_q10_average_score', 'science_q11_average_score', 'science_q12_average_score',
    'science_q13_average_score', 'science_q14_average_score', 'science_q15_average_score',
    'science_q16_average_score', 'science_q17_average_score', 'science_q18_average_score',
    'science_q19_average_score',
    # Reading total timing
    'reading_q1_total_timing', 'reading_q2_total_timing', 'reading_q3_total_timing',
    'reading_q4_total_timing', 'reading_q5_total_timing', 'reading_q6_total_timing',
    'reading_q7_total_timing', 'reading_q8_total_timing', 'reading_q9_total_timing',
    'reading_q10_total_timing', 'reading_q11_total_timing', 'reading_q12_total_timing',
    'reading_q13_total_timing', 'reading_q14_total_timing', 'reading_q15_total_timing',
    # Math total timing
    'math_q1_total_timing', 'math_q2_total_timing', 'math_q3_total_timing',
    'math_q4_total_timing', 'math_q5_total_timing', 'math_q6_total_timing',
    'math_q7_total_timing', 'math_q8_total_timing', 'math_q9_total_timing',
    'math_q10_total_timing', 'math_q11_total_timing', 'math_q12_total_timing',
    'math_q13_total_timing', 'math_q14_total_timing', 'math_q15_total_timing',
    'math_q16_total_timing', 'math_q17_total_timing', 'math_q18_total_timing',
    'math_q19_total_timing', 'math_q20_total_timing', 'math_q21_total_timing',
    # Science total timing
    'science_q1_total_timing', 'science_q2_total_timing', 'science_q3_total_timing',
    'science_q4_total_timing', 'science_q5_total_timing', 'science_q6_total_timing',
    'science_q7_total_timing', 'science_q8_total_timing', 'science_q9_total_timing',
    'science_q10_total_timing', 'science_q11_total_timing', 'science_q12_total_timing',
    'science_q13_total_timing', 'science_q14_total_timing', 'science_q15_total_timing',
    'science_q16_total_timing', 'science_q17_total_timing', 'science_q18_total_timing',
    'science_q19_total_timing',
]

TIMING_COLUMNS = [c for c in COLUMNS_TO_LOAD if c.endswith("_total_timing")]


def build_config() -> PipelineConfig:
    preprocessing = PreprocessingConfig(
        columns=COLUMNS_TO_LOAD,
        winsorize_columns=TIMING_COLUMNS,
        log_columns=TIMING_COLUMNS,
        winsor_quantile=0.999,
        enforce_non_negative=True,
        max_missing_per_row=100,
    )
    architecture = ArchitectureConfig(encoder_layers=[256, 128, 64], decoder_layers=[128, 256])
    dae_cfg = DAETrainingConfig()
    reg_cfg = RegressionTrainingConfig()
    wandb_cfg = WandbConfig(project="ml_project_embedding", group_prefix="dae", run_prefix="dae")
    return PipelineConfig(
        data_dir="../../data",
        seed=42,
        run_kfold=True,
        target_column="MathScore",
        dae_group_name="dae-pretraining",
        finetune_group_name="finetune-regression",
        dae_run_name="dae-pretrain",
        regression_run_name="regression-finetune",
        embedding_prefix="embedding",
        preprocessing=preprocessing,
        architecture=architecture,
        dae=dae_cfg,
        regression=reg_cfg,
        wandb=wandb_cfg,
    )


def main():
    cfg = build_config()
    runner = PipelineRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
