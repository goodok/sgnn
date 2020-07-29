"""
Runs a model on a single node across multiple gpus.
"""
import os
import sys
import numpy as np
import torch
import hydra
import warnings
import time

import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger

from model_lightning import LightningTemplateModel
from utils import dict_flatten, watermark, log_text_as_artifact


SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


#
# https://hydra.cc/docs/next/upgrades/0.11_to_1.0/config_path_changes/

@hydra.main(config_path="./configs", config_name='example_01.yaml')
def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:

    """

    # 0 INIT TRACKER
    # https://docs.neptune.ai/integrations/pytorch_lightning.html
    try:
        import neptune
        NEPTUNE_AVAILABLE = True
    except ImportError:  # pragma: no-cover
        NEPTUNE_AVAILABLE = False

    if NEPTUNE_AVAILABLE:
        neptune_params = hparams.tracker.neptune
        if neptune_params.fn_token is not None:
            with open(os.path.expanduser(neptune_params.fn_token), 'r') as f:
                token = f.readline().splitlines()[0]
                os.environ['NEPTUNE_API_TOKEN'] = token

        hparams_flatten = dict_flatten(hparams, sep='.')
        experiment_name = hparams.tracker.get('experiment_name', None)
        tags = hparams.tracker.get('tags', '').split('/')
        offline_mode = hparams.tracker.get('offline', False)

        tracker = NeptuneLogger(
            project_name=neptune_params.project_name,
            experiment_name=experiment_name,
            params=hparams_flatten,
            tags=tags,
            offline_mode=offline_mode,
        )
    else:
        tracker = None
        warnings.warn('You want to use `neptune` logger which is not installed yet,'
                      ' install it with `pip install neptune-client`.', UserWarning)
        time.sleep(5)

    # log
    if tracker is not None:
        watermark_s = watermark(packages=['python', 'nvidia', 'cudnn', 'hostname', 'torch', 'sparseconvnet', 'pytorch-lightning', 'hydra-core'])
        log_text_as_artifact(tracker, watermark_s, "versions.txt")
        # arguments_of_script
        sysargs_s = str(sys.argv[1:])
        log_text_as_artifact(tracker, sysargs_s, "arguments_of_script.txt")

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningTemplateModel(hparams)

    if tracker is not None:
        s = str(model)
        log_text_as_artifact(tracker, s, "model_summary.txt")

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        max_epochs=hparams.train.max_epochs,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        precision=16 if hparams.use_16bit else 32,
        logger=tracker,
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    main()
