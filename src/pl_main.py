from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from utils.widerface import WIDERDetection, detection_collate
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from trains.cspdet import CspDetLoss
from models.model import load_model
from torch.backends import cudnn
cudnn.benchmark=True
class LitMNIST(LightningModule):

    def __init__(self, opt):
        super().__init__()
        print(opt.load_model)
        self.model = create_model(opt.arch, opt.heads, opt.head_conv,pretrained=opt.load_model)
        self.count = 0
        self.loss_obj = CspDetLoss(opt)

    def forward(self, x):
        x = self.model(x)
        
        return x

    def training_step(self, batch, batch_idx):
        x = batch['input']
        outputs = self(x)
        loss, loss_stats = self.loss_obj(outputs, batch)
        self.log('loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch['input']
        outputs = self(x)
        if isinstance(outputs, list):
            outputs = outputs[0]
        loss, loss_stats = self.loss_obj(outputs, batch)
        self.log('loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(),
        #                             lr=opt.lr,
        #                             momentum=0.9)
        optimizer = torch.optim.Adam(self.parameters(), opt.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            threshold=1e-5,
            threshold_mode='abs',
            factor=opt.lr_factor,
            patience=5,
            min_lr=8e-6,
            verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'loss'}


class MyDataModule(LightningDataModule):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self, stage):
        opt = self.opt
        # called on every GPU
        Dataset = get_dataset(opt.dataset, opt.task)
        self.val_data = Dataset(opt, 'val')
        self.train_data = Dataset(opt, 'train')

    def train_dataloader(self):
        opt = self.opt
        train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_data,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
        )
        return val_loader

    def test_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_data,
            batch_size=1,
            shuffle=False,
            num_workers=10,
            pin_memory=False,
        )
        return val_loader


def main(opt):
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    model = LitMNIST(opt=opt)
    dataloader = MyDataModule(opt)
    trainer = Trainer(max_epochs=200,gpus=2,accelerator='ddp',precision=16)
    trainer.fit(model, datamodule=dataloader)


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
