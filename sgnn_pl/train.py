"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser

import hydra
from omegaconf import DictConfig

import numpy as np
import torch

import pytorch_lightning as pl
from model_lightning import LightningTemplateModel

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
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    ## ------------------------
    ## TRAINING ARGUMENTS
    ## ------------------------
    ## these are project-wide arguments

    #root_dir = os.path.dirname(os.path.realpath(__file__))
    #parent_parser = ArgumentParser(add_help=False)

    ## gpu args
    #parent_parser.add_argument(
        #'--gpus',
        #type=int,
        #default=2,
        #help='how many gpus'
    #)
    #parent_parser.add_argument(
        #'--distributed_backend',
        #type=str,
        #default='dp',
        #help='supports three options dp, ddp, ddp2'
    #)
    #parent_parser.add_argument(
        #'--use_16bit',
        #dest='use_16bit',
        #action='store_true',
        #help='if true uses 16 bit precision'
    #)

    ## each LightningModule defines arguments relevant to it
    #parser = LightningTemplateModel.add_model_specific_args(parent_parser, root_dir)
    #hyperparams = parser.parse_args()

    ## ---------------------
    ## RUN TRAINING
    ## ---------------------
    #main(hyperparams)

    main()
