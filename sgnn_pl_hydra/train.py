"""
Runs a model on a single node across multiple gpus.
"""
import os
import numpy as np
import torch
import hydra

import pytorch_lightning as pl
from pytorch_lightning.logging.neptune import NeptuneLogger

from model_lightning import LightningTemplateModel
from utils import dict_flatten

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
    neptune_params = hparams.tracker.neptune
    if neptune_params.fn_token is not None:
        with open(os.path.expanduser(neptune_params.fn_token), 'r') as f:
            token = f.readline().splitlines()[0]
            os.environ['NEPTUNE_API_TOKEN'] = token

    hparams_flatten = dict_flatten(hparams, sep='.')
    experiment_name = hparams.tracker.get('experiment_name', None)

    neptune_logger = NeptuneLogger(
        project_name=neptune_params.project_name,
        experiment_name=experiment_name,
        params=hparams_flatten,
    )

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningTemplateModel(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------

    print("hparams.gpus:", hparams.gpus)

    trainer = pl.Trainer(
        max_epochs=hparams.train.max_epochs,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        precision=16 if hparams.use_16bit else 32,
        logger=neptune_logger,
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    main()
