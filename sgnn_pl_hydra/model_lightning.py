"""
Example template for defining a system.
"""
import os
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule


import scene_dataloader
import data_util
import model as sgnn_model
import loss as loss_util

from hydra import utils


class LightningTemplateModel(LightningModule):
    """
    Sample model to show how to define a template.

    Example:

        >>> # define simple Net for MNIST dataset
        >>> params = dict(
        ...     drop_prob=0.2,
        ...     batch_size=2,
        ...     in_features=28 * 28,
        ...     learning_rate=0.001 * 8,
        ...     optimizer_name='adam',
        ...     data_root='./datasets',
        ...     out_features=10,
        ...     hidden_dim=1000,
        ... )
        >>> from argparse import Namespace
        >>> hparams = Namespace(**params)
        >>> model = LightningTemplateModel(hparams)
    """

    def __init__(self, hparams):
        """
        Pass in hyperparameters as a `argparse.Namespace` or a `dict` to the model.
        """
        # init superclass
        super().__init__()
        self.hparams = hparams

        input_dim = self.hparams.model.input_dim
        if input_dim == '':
            # set default values
            input_dim = 2 ** (3 + self.hparams.train.num_hierarchy_levels)
        elif input_dim == '128-64-64':
            input_dim = (128, 64, 64)
        elif input_dim == '160-96-96':
            input_dim = (160, 96, 96)
        elif input_dim == '64-64-64':
            input_dim = (64, 64, 64)
        self.hparams.model.input_dim = input_dim


        self._iter_counter = 0

        # if you specify an example input, the summary will show input/output for each layer
        #self.example_input_array = torch.rand(5, 28 * 28)

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout the model.
        """
        input_dim = tuple(self.hparams.model.input_dim)

        model = sgnn_model.GenModel(self.hparams.model.encoder_dim,
                               input_dim,
                               self.hparams.train.input_nf,
                               self.hparams.model.coarse_feat_dim,
                               self.hparams.model.refine_feat_dim,
                               self.hparams.train.num_hierarchy_levels,
                               not self.hparams.model.no_pass_occ,
                               not self.hparams.model.no_pass_feats,
                               self.hparams.model.use_skip_sparse,
                               self.hparams.model.use_skip_dense)

        self.model = model.cuda()

    def summarize(self, mode=None):
        return None

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x, loss_weights):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        #x = self.c_d1(x)
        #x = torch.tanh(x)
        #x = self.c_d1_bn(x)
        #x = self.c_d1_drop(x)

        #x = self.c_d2(x)
        #logits = F.log_softmax(x, dim=1)

        output = self.model(x, loss_weights)

        return output

    def loss(self, labels, logits):
        nll = F.nll_loss(logits, labels)
        return nll

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        ## forward pass
        #x, y = batch
        #x = x.view(x.size(0), -1)

        #y_hat = self(x)

        ## calculate loss
        #loss_val = self.loss(y, y_hat)

        #tqdm_dict = {'train_loss': loss_val}
        #output = OrderedDict({
            #'loss': loss_val,
            #'progress_bar': tqdm_dict,
            #'log': tqdm_dict
        #})

        ## can also return just a scalar instead of a dict (return loss_val)
        #return output

        batch_size = self.hparams.train.batch_size
        num_hierarchy_levels = self.hparams.train.num_hierarchy_levels
        truncation = self.hparams.train.truncation
        use_loss_masking = self.hparams.train.use_loss_masking
        logweight_target_sdf = self.hparams.model.logweight_target_sdf
        weight_missing_geo = self.hparams.train.weight_missing_geo


        sample = batch

        sdfs = sample['sdf']
        # TODO: fix it
        #if sdfs.shape[0] < batch_size:
        #    continue  # maintain same batch size for training
        inputs = sample['input']
        known = sample['known']
        hierarchy = sample['hierarchy']
        for h in range(len(hierarchy)):
            hierarchy[h] = hierarchy[h].cuda()
        if use_loss_masking:
            known = known.cuda()
        inputs[0] = inputs[0].cuda()
        inputs[1] = inputs[1].cuda()
        target_for_sdf, target_for_occs, target_for_hier = loss_util.compute_targets(sdfs.cuda(), hierarchy, num_hierarchy_levels, truncation, use_loss_masking, known)

        # TODO: update
        #loss_weights = self.model._loss_weights
        _iter = self._iter_counter
        loss_weights = get_loss_weights(_iter,
                                        self.hparams.train.num_hierarchy_levels,
                                        self.hparams.train.num_iters_per_level,
                                        self.hparams.train.weight_sdf_loss)


        output_sdf, output_occs = self(inputs, loss_weights)
        loss, losses = loss_util.compute_loss(output_sdf, output_occs, target_for_sdf, target_for_occs, target_for_hier, loss_weights, truncation,
                                              logweight_target_sdf, weight_missing_geo, inputs[0], use_loss_masking, known)

        tqdm_dict = {'train_loss': loss}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        self._iter_counter += 1

        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        #x, y = batch
        #x = x.view(x.size(0), -1)
        #y_hat = self(x)

        #loss_val = self.loss(y, y_hat)

        ## acc
        #labels_hat = torch.argmax(y_hat, dim=1)
        #val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        #val_acc = torch.tensor(val_acc)

        #if self.on_gpu:
            #val_acc = val_acc.cuda(loss_val.device.index)

        #output = OrderedDict({
            #'val_loss': loss_val,
            #'val_acc': val_acc,
        #})

        ## can also return just a scalar instead of a dict (return loss_val)
        #return output

        batch_size = self.hparams.train.batch_size
        num_hierarchy_levels = self.hparams.train.num_hierarchy_levels
        truncation = self.hparams.train.truncation
        use_loss_masking = self.hparams.train.use_loss_masking
        logweight_target_sdf = self.hparams.model.logweight_target_sdf
        weight_missing_geo = self.hparams.train.weight_missing_geo

        sample = batch

        sdfs = sample['sdf']
        # TODO: fix it
        #if sdfs.shape[0] < batch_size:
        #    continue  # maintain same batch size for training
        inputs = sample['input']
        known = sample['known']
        hierarchy = sample['hierarchy']
        for h in range(len(hierarchy)):
            hierarchy[h] = hierarchy[h].cuda()
        if use_loss_masking:
            known = known.cuda()
        inputs[0] = inputs[0].cuda()
        inputs[1] = inputs[1].cuda()
        target_for_sdf, target_for_occs, target_for_hier = loss_util.compute_targets(sdfs.cuda(), hierarchy, num_hierarchy_levels, truncation, use_loss_masking, known)

        # TODO: update
        _iter = self._iter_counter
        loss_weights = get_loss_weights(_iter,
                                        self.hparams.train.num_hierarchy_levels,
                                        self.hparams.train.num_iters_per_level,
                                        self.hparams.train.weight_sdf_loss)

        output_sdf, output_occs = self(inputs, loss_weights)
        loss, losses = loss_util.compute_loss(output_sdf, output_occs, target_for_sdf, target_for_occs, target_for_hier, loss_weights, truncation,
                                              logweight_target_sdf, weight_missing_geo, inputs[0], use_loss_masking, known)


        output = OrderedDict({
            'val_loss': loss,
        })
        return output



    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

        val_loss_mean /= len(outputs)

        tqdm_dict = {'val_loss': val_loss_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        lr = self.hparams.train.lr
        weight_decay = self.hparams.train.weight_decay
        decay_lr = self.hparams.train.decay_lr
        last_epoch = -1 if not self.hparams.train.retrain else self.hparams.train.start_epoch - 1

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_lr, gamma=0.5, last_epoch=last_epoch)
        return [optimizer], [scheduler]

    #def __dataloader(self, train):
        ## this is neede when you want some info about dataset before binding to trainer
        #self.prepare_data()
        ## init data generators
        #transform = transforms.Compose([transforms.ToTensor(),
                                        #transforms.Normalize((0.5,), (1.0,))])
        #dataset = MNIST(root=self.hparams.data_root, train=train,
                        #transform=transform, download=False)

        ## when using multi-node (ddp) we need to add the  datasampler
        #batch_size = self.hparams.batch_size

        #loader = DataLoader(
            #dataset=dataset,
            #batch_size=batch_size,
            #num_workers=0
        #)

        #return loader

    #def prepare_data(self):
        #transform = transforms.Compose([transforms.ToTensor(),
                                        #transforms.Normalize((0.5,), (1.0,))])
        #_ = MNIST(root=self.hparams.data_root, train=True,
                  #transform=transform, download=True)

    #def train_dataloader(self):
        #log.info('Training data loader called.')
        #return self.__dataloader(train=True)

    #def val_dataloader(self):
        #log.info('Validation data loader called.')
        #return self.__dataloader(train=False)

    #def test_dataloader(self):
        #log.info('Test data loader called.')
        #return self.__dataloader(train=False)

    def _get_train_files(self):
        #print("CD:", pathlib.Path().absolute())
        #print("CD:", )
        root_dir = Path(utils.get_original_cwd())

        if not hasattr(self, '__data_path'):

            data_path = root_dir / self.hparams.data.data_path
            train_file_list = root_dir / self.hparams.data.train_file_list
            val_file_list = root_dir / self.hparams.data.val_file_list

            train_files, val_files = data_util.get_train_files(data_path, train_file_list, val_file_list)

            self.__data_path = data_path
            self.__train_files = train_files
            self.__val_files = val_files

        return self.__data_path, self.__train_files, self.__val_files

    def train_dataloader(self):
        log.info('Training data loader called.')

        data_path, train_files, val_files = self._get_train_files()

        input_dim = self.hparams.model.input_dim
        num_hierarchy_levels = self.hparams.train.num_hierarchy_levels
        truncation = self.hparams.train.truncation
        batch_size = self.hparams.train.batch_size
        num_workers_train = self.hparams.train.num_workers_train

        _OVERFIT = False
        if len(train_files) == 1:
            _OVERFIT = True
            # TODO:
            #args.use_loss_masking = False
        num_overfit_train = 0 if not _OVERFIT else 640
        num_overfit_val = 0 if not _OVERFIT else 160
        print('#train files = ', len(train_files))
        print('#val files = ', len(val_files))
        train_dataset = scene_dataloader.SceneDataset(train_files, input_dim, truncation, num_hierarchy_levels, 0, num_overfit_train)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_train,
                                                       collate_fn=scene_dataloader.collate)

        self._iter_counter = self.hparams.train.start_epoch * (len(train_dataset) // self.hparams.train.batch_size)


        return train_dataloader


    def val_dataloader(self):
        log.info('Validation data loader called.')

        data_path, train_files, val_files = self._get_train_files()

        input_dim = tuple(self.hparams.model.input_dim)

        num_hierarchy_levels = self.hparams.train.num_hierarchy_levels
        truncation = self.hparams.train.truncation
        batch_size = self.hparams.train.batch_size

        num_workers_valid = self.hparams.train.num_workers_valid
        num_overfit_val = self.hparams.train.num_overfit_val

        if len(val_files) > 0:
            val_dataset = scene_dataloader.SceneDataset(val_files, input_dim, truncation, num_hierarchy_levels, 0, num_overfit_val)
            print('val_dataset', len(val_dataset))
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_valid, collate_fn=scene_dataloader.collate)
        return val_dataloader


    def test_step(self, batch, batch_idx):
        """
        Lightning calls this during testing, similar to `validation_step`,
        with the data from the test dataloader passed in as `batch`.
        """
        output = self.validation_step(batch, batch_idx)
        # Rename output keys
        output['test_loss'] = output.pop('val_loss')
        output['test_acc'] = output.pop('val_acc')

        return output

    def test_epoch_end(self, outputs):
        """
        Called at the end of test to aggregate outputs, similar to `validation_epoch_end`.
        :param outputs: list of individual outputs of each test step
        """
        results = self.validation_step_end(outputs)

        # rename some keys
        results['progress_bar'].update({
            'test_loss': results['progress_bar'].pop('val_loss'),
            'test_acc': results['progress_bar'].pop('val_acc'),
        })
        results['log'] = results['progress_bar']
        results['test_loss'] = results.pop('val_loss')

        return results

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Parameters you define here will be available to your model through `self.hparams`.
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        ## network params
        #parser.add_argument('--in_features', default=28 * 28, type=int)
        #parser.add_argument('--out_features', default=10, type=int)
        ## use 500 for CPU, 50000 for GPU to see speed difference
        #parser.add_argument('--hidden_dim', default=50000, type=int)
        #parser.add_argument('--drop_prob', default=0.2, type=float)
        #parser.add_argument('--learning_rate', default=0.001, type=float)

        ## data
        #parser.add_argument('--data_root', default=os.path.join(root_dir, 'mnist'), type=str)

        ## training params (opt)
        #parser.add_argument('--epochs', default=20, type=int)
        #parser.add_argument('--optimizer_name', default='adam', type=str)
        #parser.add_argument('--batch_size', default=64, type=int)

        #parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')

        parser.add_argument('--data_path', required=True, help='path to data')
        parser.add_argument('--train_file_list', required=True, help='path to file list of train data')
        parser.add_argument('--val_file_list', default='', help='path to file list of val data')
        parser.add_argument('--save', default='./logs', help='folder to output model checkpoints')
        # model params
        parser.add_argument('--retrain', type=str, default='', help='model to load from')
        parser.add_argument('--input_dim', type=str, default='128-64-64', help='voxel dim.')
        parser.add_argument('--encoder_dim', type=int, default=8, help='pointnet feature dim')
        parser.add_argument('--coarse_feat_dim', type=int, default=16, help='feature dim')
        parser.add_argument('--refine_feat_dim', type=int, default=16, help='feature dim')
        parser.add_argument('--no_pass_occ', dest='no_pass_occ', action='store_true')
        parser.add_argument('--no_pass_feats', dest='no_pass_feats', action='store_true')
        parser.add_argument('--use_skip_sparse', type=int, default=1, help='use skip connections between sparse convs')
        parser.add_argument('--use_skip_dense', type=int, default=1, help='use skip connections between dense convs')
        parser.add_argument('--no_logweight_target_sdf', dest='logweight_target_sdf', action='store_false')
        # train params
        parser.add_argument('--num_hierarchy_levels', type=int, default=4, help='#hierarchy levels (must be > 1).')
        parser.add_argument('--num_iters_per_level', type=int, default=2000, help='#iters before fading in training for next level.')
        parser.add_argument('--num_overfit_val', type=int, default=160, help='')
        parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        parser.add_argument('--num_workers_train', type=int, default=4, help='num workers for train')
        parser.add_argument('--num_workers_valid', type=int, default=4, help='num workers for valid')
        parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
        parser.add_argument('--max_epochs', type=int, default=5, help='number of epochs to train for')
        parser.add_argument('--save_epoch', type=int, default=1, help='save every nth epoch')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
        parser.add_argument('--decay_lr', type=int, default=10, help='decay learning rate by half every n epochs')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay.')
        parser.add_argument('--weight_sdf_loss', type=float, default=1.0, help='weight sdf loss vs occ.')
        parser.add_argument('--weight_missing_geo', type=float, default=5.0, help='weight missing geometry vs rest of sdf.')
        parser.add_argument('--vis_dfs', type=int, default=0, help='use df (iso 1) to visualize')
        parser.add_argument('--use_loss_masking', dest='use_loss_masking', action='store_true')
        parser.add_argument('--no_loss_masking', dest='use_loss_masking', action='store_false')
        parser.add_argument('--scheduler_step_size', type=int, default=0, help='#iters before scheduler step (0 for each epoch)')
        parser.add_argument('--input_nf', type=int, default=1, help='from original hardcoded')

        return parser

import numpy as np


def get_loss_weights(iter, num_hierarchy_levels, num_iters_per_level, factor_l1_loss):
    weights = np.zeros(num_hierarchy_levels+1, dtype=np.float32)
    cur_level = iter // num_iters_per_level
    if cur_level > num_hierarchy_levels:
        weights.fill(1)
        weights[-1] = factor_l1_loss
        if iter == (num_hierarchy_levels + 1) * num_iters_per_level:
            print('[iter %d] updating loss weights:' % iter, weights)
        return weights
    for level in range(0, cur_level+1):
        weights[level] = 1.0
    step_factor = 20
    fade_amount = max(1.0, min(100, num_iters_per_level//step_factor))
    fade_level = iter % num_iters_per_level
    cur_weight = 0.0
    l1_weight = 0.0
    if fade_level >= num_iters_per_level - fade_amount + step_factor:
        fade_level_step = (fade_level - num_iters_per_level + fade_amount) // step_factor
        cur_weight = float(fade_level_step) / float(fade_amount//step_factor)
    if cur_level+1 < num_hierarchy_levels:
        weights[cur_level+1] = cur_weight
    elif cur_level < num_hierarchy_levels:
        l1_weight = factor_l1_loss * cur_weight
    else:
        l1_weight = 1.0
    weights[-1] = l1_weight
    if iter % num_iters_per_level == 0 or (fade_level >= num_iters_per_level - fade_amount + step_factor and (fade_level - num_iters_per_level + fade_amount) % step_factor == 0):
        print('[iter %d] updating loss weights:' % iter, weights)
    return weights
