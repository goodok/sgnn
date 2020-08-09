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
from pathlib import Path
from traceback import print_exc

import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks.lr_logger import LearningRateLogger

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

    USE_NEPTUNE = False
    if getattr(hparams, 'tracker', None) is not None:
        if getattr(hparams.tracker, 'neptune', None) is not None:
            USE_NEPTUNE = True

    if USE_NEPTUNE and not NEPTUNE_AVAILABLE:
        warnings.warn('You want to use `neptune` logger which is not installed yet,'
                      ' install it with `pip install neptune-client`.', UserWarning)
        time.sleep(5)

    tracker = None

    if NEPTUNE_AVAILABLE and USE_NEPTUNE:
        neptune_params = hparams.tracker.neptune
        fn_token = getattr(neptune_params, 'fn_token', None)
        if fn_token is not None:
            p = Path(neptune_params.fn_token).expanduser()
            if p.exists():
                with open(p, 'r') as f:
                    token = f.readline().splitlines()[0]
                    os.environ['NEPTUNE_API_TOKEN'] = token

        hparams_flatten = dict_flatten(hparams, sep='.')
        experiment_name = hparams.tracker.get('experiment_name', None)
        tags = list(hparams.tracker.get('tags', []))
        offline_mode = hparams.tracker.get('offline', False)

        tracker = NeptuneLogger(
            project_name=neptune_params.project_name,
            experiment_name=experiment_name,
            params=hparams_flatten,
            tags=tags,
            offline_mode=offline_mode,
            upload_source_files=["../../../*.py"],  # because hydra change current dir
        )

    try:

        # log
        if tracker is not None:
            watermark_s = watermark(packages=['python', 'nvidia', 'cudnn', 'hostname', 'torch', 'sparseconvnet', 'pytorch-lightning', 'hydra-core',
                                              'numpy', 'plyfile'])
            log_text_as_artifact(tracker, watermark_s, "versions.txt")
            # arguments_of_script
            sysargs_s = str(sys.argv[1:])
            log_text_as_artifact(tracker, sysargs_s, "arguments_of_script.txt")

            for key in ['overrides.yaml', 'config.yaml']:
                p = Path.cwd() / '.hydra' / key
                if p.exists():
                    tracker.log_artifact(str(p), f'hydra_{key}')


        callbacks = []
        if tracker is not None:
            lr_logger = LearningRateLogger()
            callbacks.append(lr_logger)


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
        cfg = hparams.PL

        if tracker is None:
           tracker = cfg.logger   # True by default in PL

        kwargs = dict(cfg)
        kwargs.pop('logger')

        trainer = pl.Trainer(
            max_epochs=hparams.train.max_epochs,
            callbacks=callbacks,
            logger=tracker,
            **kwargs,
        )

        # ------------------------
        # 3 START TRAINING
        # ------------------------
        print()
        print("Start training")

        trainer.fit(model)

    except (Exception, KeyboardInterrupt) as ex:
        if tracker is not None:
            print_exc()
            tracker.experiment.stop(str(ex))
        raise


if __name__ == '__main__':
    main()
