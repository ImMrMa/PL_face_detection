from .widers import config
import pickle
from .widers import data_augment
import numpy as np
import random
from .widers.bbox_transform import *


class WiderCsp():
    num_classes = 1
    default_resolution = [704, 704]
    mean = np.array([0.485, 0.456,
                              0.406]).reshape(1, 1, 3).astype(np.float32)
    std = np.array([0.229, 0.224,
                             0.225]).reshape(1, 1, 3).astype(np.float32)
    def __init__(self, opt=None,mode='train'):
        
        self.C = config.Config()
        if mode == 'train':
            cache_path = '../data/cache/widerface/train'
        elif mode == 'val':
            cache_path = '../data/cache/widerface/val'
        else:
            print('choose error opt')
        with open(cache_path, 'rb') as fid:
            self.cache_data = pickle.load(fid, encoding='latin1')
        num_imgs_train = len(self.cache_data)
        print('num_imgs:', num_imgs_train)

    def __len__(self):
        return len(self.cache_data)

    def __getitem__(self, index):
        C = self.C
        img_data, img = data_augment.augment_wider(self.cache_data[index], C)
        hm, wh, offset = self.calc_gt_center(C,
                                             img_data,
                                             down=C.down,
                                             scale=C.scale,
                                             offset=True)
        img = img.astype(np.float32)
        img = img / 255
        img = img[..., [2, 1, 0]]
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        hm = hm.transpose(2, 0, 1).astype(np.float32)
        wh = wh.transpose(2, 0, 1).astype(np.float32)
        offset = offset.transpose(2, 0, 1).astype(np.float32)
        return dict(input=img, hm=hm, wh=wh, offset=offset)

    def calc_gt_center(self, C, img_data, r=2, down=4, scale='h', offset=True):
        def gaussian(kernel):
            sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
            s = 2 * (sigma**2)
            dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
            return np.reshape(dx, (-1, 1))

        gts = np.copy(img_data['bboxes'])
        igs = np.copy(img_data['ignoreareas'])
        scale_map = np.zeros(
            (int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
        if scale == 'hw':
            scale_map = np.zeros(
                (int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
        if offset:
            offset_map = np.zeros(
                (int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
        seman_map = np.zeros(
            (int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
        seman_map[:, :, 1] = 1
        if len(igs) > 0:
            igs = igs / down
            for ind in range(len(igs)):
                x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(
                    np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
                seman_map[y1:y2, x1:x2, 1] = 0
        if len(gts) > 0:
            gts = gts / down
            for ind in range(len(gts)):
                # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
                x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(
                    np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
                c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int(
                    (gts[ind, 1] + gts[ind, 3]) / 2)
                dx = gaussian(x2 - x1)
                dy = gaussian(y2 - y1)
                gau_map = np.multiply(dy, np.transpose(dx))
                seman_map[y1:y2, x1:x2, 0] = np.maximum(
                    seman_map[y1:y2, x1:x2, 0], gau_map)
                seman_map[y1:y2, x1:x2, 1] = 1
                seman_map[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r +
                              1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1
                elif scale == 'w':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r +
                              1, 0] = np.log(gts[ind, 2] - gts[ind, 0])
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1
                elif scale == 'hw':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r +
                              1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r +
                              1, 1] = np.log(gts[ind, 2] - gts[ind, 0])
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 2] = 1
                if offset:
                    offset_map[c_y, c_x, 0] = (gts[ind, 1] +
                                               gts[ind, 3]) / 2 - c_y - 0.5
                    offset_map[c_y, c_x, 1] = (gts[ind, 0] +
                                               gts[ind, 2]) / 2 - c_x - 0.5
                    offset_map[c_y, c_x, 2] = 1

        if offset:
            return seman_map, scale_map, offset_map
        else:
            return seman_map, scale_map