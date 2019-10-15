from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

# from external.nms import soft_nms
from models.decode import fadet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger
from PIL import Image
from .base_detector import BaseDetector
from py_cpu_nms import py_cpu_nms as nms
class FadetDetector(BaseDetector):
    def __init__(self, opt):
        super(FadetDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        # def add_box(image, bbox, cat_id):
        #     bbox = [int(i) for i in bbox]
        #     cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
        #                   (255, 0, 0), 2)
        #     return image

        with torch.no_grad():
            output = self.model(images)[-1]
            hm_small = output['hm_small'].sigmoid_()
            wh_small = output['wh_small'].sigmoid_()
            hm_norm = output['hm_norm'].sigmoid_()
            wh_norm = output['wh_norm'].sigmoid_()
            del output
            reg = output['reg'] if self.opt.reg_offset else None
            if self.opt.flip_test:
                hm_small = (hm_small[0:1] + flip_tensor(hm_small[1:2])) / 2
                wh_small = (wh_small[0:1] + flip_tensor(wh_small[1:2])) / 2
                hm_norm = (hm_norm[0:1] + flip_tensor(hm_norm[1:2])) / 2
                wh_norm = (wh_norm[0:1] + flip_tensor(wh_norm[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
            torch.cuda.synchronize()
            
            forward_time = time.time()
            torch.cuda.empty_cache()
            dets = fadet_decode(hm_small,
                                wh_small,
                                reg=reg,
                                p=0.15,
                                large=False)
            dets2 = fadet_decode(hm_norm, wh_norm, reg=reg, p=0.2, large=True)
            dets = torch.cat([dets, dets2], dim=1)
            # img = images[0]

            # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
            # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
            # img = self.hhh_image
            # j = Image.fromarray(img)
            # j.save('img.jpg')
            # print(img.shape)
            # img_box = img.copy()
            # for x1, y1, x2, y2, _, _ in dets.reshape(-1, 6):
            #     add_box(img_box, [x1, y1, x2, y2], 0)
            # j = Image.fromarray(img_box)
            # j.save('img_box.jpg')
            # print((dets.shape))
            # input('s')
        if return_time:
            return  dets, forward_time
        else:
            return  dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']],
                                  meta['out_height'], meta['out_width'],
                                  self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections],
                axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.opt.nms:
                keep=nms(results[j], thresh=0.5)
                results[j]=results[j][keep,:]
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(
                output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, 4] > self.opt.center_thresh:
                    debugger.add_coco_bbox(
                        detection[i, k, :4],
                        detection[i, k, -1],
                        detection[i, k, 4],
                        img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4],
                                           j - 1,
                                           bbox[4],
                                           img_id='ctdet')
        debugger.show_all_imgs(pause=self.pause)
