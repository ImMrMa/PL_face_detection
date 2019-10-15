from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegFaL1Loss,WhFaL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer


class FadetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(FadetLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegFaL1Loss()
        self.crit_wh = WhFaL1Loss()
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        hm_small_loss,hm_norm_loss, wh_small_loss,wh_norm_loss,off_small_loss,off_norm_loss = 0, 0,0,0,0,0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm_small']=_sigmoid(output['hm_small'])
                output['hm_norm']=_sigmoid(output['hm_norm'])
            if opt.eval_oracle_hm:
                output['hm'] = batch['hm']
            if opt.eval_oracle_wh:
                output['wh'] = torch.from_numpy(gen_oracle_map(
                    batch['wh'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
            if opt.eval_oracle_offset:
                output['reg'] = torch.from_numpy(gen_oracle_map(
                    batch['reg'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)
            hm_norm_loss += self.crit(output['hm_norm'], batch['hm_norm']) / opt.num_stacks
            hm_small_loss += self.crit(output['hm_small'],batch['hm_small'])/opt.num_stacks
            if opt.wh_weight > 0:
                output['wh_small']=_sigmoid(output['wh_small'])
                output['wh_norm']=_sigmoid(output['wh_norm'])
                wh_small_loss += self.crit_wh(output['wh_small'],
                                          batch['mask_small'], batch['wh_small']) / opt.num_stacks
                wh_norm_loss += self.crit_wh(output['wh_norm'],
                                          batch['mask_norm'], batch['wh_norm']) / opt.num_stacks
            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'],
                                          batch['mask'], batch['offset']) / opt.num_stacks
        hm_loss=hm_norm_loss+hm_small_loss
        wh_loss=wh_norm_loss+wh_small_loss
        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss 
        # loss =opt.wh_weight * wh_loss 
        loss_stats = {'loss': loss, 'hm_s': hm_small_loss,'hm_n':hm_norm_loss,'wh_s':wh_small_loss,'wh_n':wh_norm_loss }
        if opt.reg_offset:
            loss_stats['off_loss'] = off_loss
        return loss, loss_stats


class FadetTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(FadetTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_s', 'hm_n','wh_s','wh_n']
        if opt.reg_offset:
            loss_states.append('off_loss')
        loss = FadetLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=opt.cat_spec_wh, K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        for i in range(1):
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
                img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(
                output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                           dets[i, k, 4], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                           dets_gt[i, k, 4], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(
                    opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
