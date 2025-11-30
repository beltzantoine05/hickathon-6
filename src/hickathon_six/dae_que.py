"""Configuration preset for the QUE feature set pipeline."""
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
    'WB153', 'ST253', 'ST311', 'WB165', 'IC171', 'ST038', 'IC180', 'ST255', 'ST350', 'ST034',
    'FL162', 'PA195', 'ST330', 'FL169', 'ST230', 'ST331', 'ST354', 'PA042', 'ST305', 'ST258',
    'PA167', 'IC176', 'ST352', 'IC173', 'PA188', 'ST283', 'ST256', 'WB154', 'WB166', 'FL150',
    'ST266', 'IC174', 'ST005', 'ST021', 'IC182', 'FL164', 'FL160', 'ST267', 'ST338', 'ST351',
    'ST342', 'ST355', 'FL166', 'PA183', 'ST007', 'FL170', 'WB168', 'WB163', 'IC184', 'ST348',
    'ST226', 'ST336', 'WB160', 'FL167', 'PA166', 'ST250', 'ST268', 'IC177', 'PA196', 'IC183',
    'ST275', 'IC172', 'IC175', 'ST340', 'ST296', 'ST270', 'ST289', 'ST353', 'ST349', 'ST301',
    'PA186', 'WB164', 'PA197', 'ST290', 'ST062', 'ST254', 'ST345', 'ST322', 'IC170', 'PA006',
    'ST347', 'ST343', 'ST307', 'ST160', 'ST168', 'ST161', 'ST153', 'ST188', 'PA009', 'PA182',
    'PA004', 'ST059', 'ST150', 'PA158', 'WB155', 'WB162', 'ST127', 'EC162', 'ST104', 'PA003',
    'ST125', 'ST163', 'ST297', 'ST006', 'ST263', 'PA162', 'ST164', 'ST008', 'EC031', 'WB177',
    'ST183', 'EC012', 'ST223', 'ST177', 'ST102', 'PA041', 'EC163', 'PA008', 'PA177', 'ST175',
    'ST273', 'EC153', 'ST212', 'PA154', 'PA156', 'PA007', 'PA159', 'ST300', 'ST036', 'PA018',
    'ST208', 'ST165', 'PA175', 'ST167', 'ST211', 'WB176', 'ST158', 'WB178', 'IC014', 'ST213',
    'ST327', 'ST251', 'ST260', 'PA160', 'ST095', 'PA033', 'ST146', 'ST098', 'ST100', 'ST097',
    'ST113', 'PA002', 'PA032', 'IC001',
]

WINSOR_COLUMNS = [
    'WB153', 'ST253', 'WB165', 'ST038', 'ST350', 'ST034', 'FL162', 'PA195', 'FL169', 'PA042',
    'ST305', 'ST258', 'PA167', 'ST352', 'IC173', 'PA188', 'ST256', 'WB154', 'WB166', 'FL150',
    'ST266', 'IC174', 'ST005', 'ST021', 'FL164', 'FL160', 'ST267', 'ST338', 'FL166', 'PA183',
    'ST007', 'FL170', 'WB168', 'WB163', 'IC184', 'WB160', 'FL167', 'PA166', 'IC177', 'PA196',
    'ST275', 'IC175', 'ST296', 'ST353', 'ST349', 'ST301', 'PA186', 'WB164', 'PA197', 'ST062',
    'PA006', 'ST343', 'ST307', 'ST160', 'ST161', 'ST188', 'PA009', 'PA182', 'PA004', 'ST059',
    'PA158', 'WB155', 'WB162', 'ST127', 'EC162', 'PA003', 'ST297', 'ST006', 'PA162', 'ST008',
    'EC031', 'WB177', 'EC012', 'ST223', 'PA041', 'EC163', 'PA008', 'PA177', 'ST273', 'PA154',
    'PA156', 'PA007', 'PA159', 'ST036', 'PA018', 'PA175', 'WB176', 'WB178', 'ST260', 'PA160',
    'PA033', 'ST146', 'PA002', 'PA032',
]



def build_config() -> PipelineConfig:
    preprocessing = PreprocessingConfig(
        columns=COLUMNS_TO_LOAD,
        winsorize_columns=WINSOR_COLUMNS,
        log_columns=[],
        winsor_quantile=0.999,
        enforce_non_negative=True,
        max_missing_per_row=100,
    )
    architecture = ArchitectureConfig(encoder_layers=[512, 288, 64], decoder_layers=[288, 512])
    dae_cfg = DAETrainingConfig()
    reg_cfg = RegressionTrainingConfig()
    wandb_cfg = WandbConfig(project="hi6", group_prefix="que", run_prefix="que", entity="themlaw-personal")
    return PipelineConfig(
        data_dir="../../data",
        seed=42,
        run_kfold=True,
        target_column="MathScore",
        dae_group_name="que-dae-pretraining",
        finetune_group_name="que-finetune-regression",
        dae_run_name="que-dae-pretrain",
        regression_run_name="que-regression-finetune",
        embedding_prefix="que_embedding",
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
