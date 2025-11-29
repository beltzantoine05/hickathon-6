from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import wandb

from hickathon_six.DAE.config import Config


def run_name(prefix: str, label: str) -> str:
    return f"{prefix}-{label}"


@contextmanager
def wandb_run(cfg: Config, name: str, group: Optional[str] = None, job_type: Optional[str] = None):
    init_kwargs = {
        "project": cfg.wandb_project,
        "name": name,
        "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__.keys()},
        "tags": cfg.wandb_tags,
    }
    if cfg.wandb_entity:
        init_kwargs["entity"] = cfg.wandb_entity
    if group:
        init_kwargs["group"] = group
    if job_type:
        init_kwargs["job_type"] = job_type
    if cfg.wandb_mode:
        init_kwargs["mode"] = cfg.wandb_mode

    run = wandb.init(**init_kwargs)
    try:
        yield run
    finally:
        run.finish()


def log_metrics(run: wandb.sdk.wandb_run.Run, metrics: dict[str, float], step: Optional[int] = None):
    if step is None:
        run.log(metrics)
    else:
        run.log(metrics, step=step)
