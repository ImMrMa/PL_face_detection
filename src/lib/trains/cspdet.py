from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.losses import FocalLoss
from models.losses import RegFaL1Loss, WhFaL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer


class HmLoss(torch.nn.Module):
    def __init__(self):
        super(HmLoss, self).__init__()
        self.bce_loss = F.binary_cross_entropy

    def forward(self, output, label):
        cls_loss = self.bce_loss(output[:, 0, ...],
                                 label[:, 2, ...],
                                 reduction='none')
        positives = label[:, 2, :, :]
        negatives = label[:, 1, :, :] - label[:, 2, :, :]
        foreground_weight = positives * ((1.0 - output[:, 0, :, :])**2)
        background_weight = negatives * (
            (1.0 - label[:, 0, :, :])**4.0) * (output[:, 0, :, :]**2)
        focal_weight = foreground_weight + background_weight
        loss = torch.sum(focal_weight * cls_loss) / torch.clamp(
            torch.sum(label[:, 2, :, :]), 1)
        return loss
class MaskLoss(torch.nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
        self.bce_loss = F.binary_cross_entropy

    def forward(self, output, label):
        cls_loss = self.bce_loss(output[:, 0, ...],
                                 label[:, 0, ...],
                                 reduction='none')
        positives = label[:, 0, :, :]
        negatives = label[:, 0, :, :]==0
        foreground_weight = positives * ((1.0 - output[:, 0, :, :])**2)
        background_weight = negatives * (
            (1.0 - label[:, 0, :, :])**4.0) * (output[:, 0, :, :]**2)
        focal_weight = foreground_weight + background_weight
        loss = torch.sum(focal_weight * cls_loss) / torch.clamp(
            torch.sum(label[:, 2, :, :]), 1)
        return loss


class WhLoss(torch.nn.Module):
    def __init__(self):
        super(WhLoss, self).__init__()
        self.sl1_loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, output, label):
        abs_loss = torch.abs(label[:, :2, ...] - output) / (label[:, :2, ...] +
                                                            1e-10)
        squ_loss = 0.5 * (((label[:, :2, ...] - output) /
                           (label[:, :2, ...] + 1e-10))**2)
        loss = label[:, 2, ...] * torch.sum(
            torch.where(abs_loss < 1, squ_loss, abs_loss), 1)
        hm_loss = torch.sum(loss) / torch.clamp(torch.sum(label[:, 2, ...]), 1)
        return hm_loss


class OffsetLoss(torch.nn.Module):
    def __init__(self):
        super(OffsetLoss, self).__init__()
        self.sl1_loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, output, label):
        mask = label[:, 2:3, ...]
        hm_loss = self.sl1_loss(output * mask, label[:, :2, ...] * mask)

        hm_loss = hm_loss / torch.clamp(torch.sum(mask), 1)
        return hm_loss


class CspDetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CspDetLoss, self).__init__()
        if opt.multi_scale:
            self.crit_small_hm=HmLoss()
            self.crit_small_wh=WhLoss()
        self.crit_hm = HmLoss()
        self.crit_off = OffsetLoss()
        self.crit_wh = WhLoss()
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        loss=0
        hm_loss, wh_loss, off_loss = 0, 0, 0
        hm_loss += self.crit_hm(outputs['hm'], batch['hm'])
        if opt.multi_scale:
            hm_small_loss=self.crit_small_hm(outputs['hm_small'],batch['hm_small'])
        if opt.wh_weight > 0:
            wh_loss += self.crit_wh(outputs['wh'], batch['wh'])
            if opt.multi_scale:
                wh_small_loss=self.crit_small_wh(outputs['wh_small'],batch['wh_small'])
        if opt.reg_offset and opt.off_weight > 0:
            off_loss += self.crit_off(outputs['offset'], batch['offset'])
        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        if opt.multi_scale:
            loss+=opt.hm_weight*hm_small_loss+opt.wh_weight*wh_small_loss
        # loss_stats=dict(loss=loss)
        loss_stats = {
            'loss': loss,
            'hm_loss': hm_loss,
            'wh_loss': wh_loss,
            'off_loss': off_loss
        }
        if opt.multi_scale:
            loss_stats['hm_small_loss']=hm_small_loss
            loss_stats['wh_small_loss']=wh_small_loss
        return loss, loss_stats


class CspDetTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(CspDetTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states=['loss']
        loss_states = ['loss', 'hm_loss', 'wh_loss']
        if opt.reg_offset:
            loss_states.append('off_loss')
        if opt.multi_scale:

            loss_states.append('hm_small_loss')
            loss_states.append('wh_small_loss')
        loss = CspDetLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        dets = ctdet_decode(output['hm'],
                            output['wh'],
                            reg=reg,
                            cat_spec_wh=opt.cat_spec_wh,
                            K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        for i in range(1):
            debugger = Debugger(dataset=opt.dataset,
                                ipynb=(opt.debug == 3),
                                theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((img * opt.std + opt.mean) * 255.), 0,
                          255).astype(np.uint8)
            pred = debugger.gen_colormap(
                output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4],
                                           dets[i, k, -1],
                                           dets[i, k, 4],
                                           img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4],
                                           dets_gt[i, k, -1],
                                           dets_gt[i, k, 4],
                                           img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir,
                                       prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = ctdet_decode(output['hm'],
                            output['wh'],
                            reg=reg,
                            cat_spec_wh=self.opt.cat_spec_wh,
                            K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(dets.copy(),
                                      batch['meta']['c'].cpu().numpy(),
                                      batch['meta']['s'].cpu().numpy(),
                                      output['hm'].shape[2],
                                      output['hm'].shape[3],
                                      output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
