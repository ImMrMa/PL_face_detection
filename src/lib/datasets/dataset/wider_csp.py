from .widers import config
import pickle
from .widers import data_augment
import numpy as np
import random
from .widers.bbox_transform import *
import cv2


class WiderCsp():
    num_classes = 1
    default_resolution = [704, 704]
    mean = np.array([0.485, 0.456,
                              0.406]).reshape(1, 1, 3).astype(np.float32)
    std = np.array([0.229, 0.224,
                             0.225]).reshape(1, 1, 3).astype(np.float32)

    def __init__(self, opt=None, mode='train'):

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
        # hm_small, wh_small, = self.calc_gt_center(C,
        #                                      img_data,
        #                                      down=1,
        #                                      scale='hw',
        #                                      offset=False,r=0)
        hm, wh, offset ,mask= self.calc_gt_center(C,
                                             img_data,
                                             down=4,
                                             scale='hw',
                                             offset=True, r=2,mask=True)

        img = img.astype(np.float32)
        img = img / 255
        img = img[..., [2, 1, 0]]
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        # hm_small = hm_small.transpose(2,0,1).astype(np.float32)
        # wh_small = wh_small.transpose(2,0,1).astype(np.float32)
        hm = hm.transpose(2, 0, 1).astype(np.float32)
        wh = wh.transpose(2, 0, 1).astype(np.float32)
        offset = offset.transpose(2, 0, 1).astype(np.float32)
        mask=mask.transpose(2,0,1).astype(np.float32)
        # ,hm_small=hm_small,wh_small=wh_small)
        return_data=dict(input=img, hm=hm, wh=wh, offset=offset,mask=mask)
        return return_data

    def calc_gt_center(self, C, img_data, r=2, down=4, scale='h', offset=True, mask=False):
        def gaussian(kernel):
            sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
            s = 2 * (sigma**2)
            dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
            return np.reshape(dx, (-1, 1))

        def draw_mask(img, bbox):
            wh = (int(np.ceil(bbox[2]-bbox[0])), int(np.ceil(bbox[3]-bbox[1])))
            center = (int((bbox[2]+bbox[0])/2), int((bbox[3]+bbox[1]/2)))
            cv2.ellipse(img, center, wh, 0, 0, 360, 1, -1)
            
        gts= np.copy(img_data['bboxes'])
        
        if down == 4:
            pass
            # ig_length=16
            # delete_indexs=[]
            # for index,box in enumerate(gts):
            #     length=((box[3]-box[1])*(box[2]-box[0]))**0.5
            #     if length<=ig_length:
            #         delete_indexs.append(index)
        elif down == 1:
            ig_length = 20
            delete_indexs = []
            for index, box in enumerate(gts):
                length = ((box[3]-box[1])*(box[2]-box[0]))**0.5
                if length > ig_length:
                    delete_indexs.append(index)
        # gts=np.delete(gts,delete_indexs,0)
        igs= np.copy(img_data['ignoreareas'])
        scale_map= np.zeros(
            (int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
        if mask:
            mask_map= np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 1))
        if scale == 'hw':
            scale_map= np.zeros(
                (int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
        if offset:
            offset_map= np.zeros(
                (int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
        seman_map= np.zeros(
            (int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
        seman_map[:, :, 1]= 1
        if len(igs) > 0:
            igs= igs / down
            for ind in range(len(igs)):
                x1, y1, x2, y2= int(igs[ind, 0]), int(igs[ind, 1]), int(
                    np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
                seman_map[y1:y2, x1:x2, 1]= 0
        if len(gts) > 0:
            gts= gts / down
            for ind in range(len(gts)):
                w, h = gts[ind, 2] - gts[ind, 0], gts[ind, 3] - gts[ind, 1]
                if w<=4 or h<=4:
                    draw_mask(mask_map,gts[ind])
                    if w<3 or h<3:
                        continue
                # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
                x1, y1, x2, y2= int(np.ceil(gts[ind, 0])), int(
                    np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
                c_x, c_y= int((gts[ind, 0] + gts[ind, 2]) / 2), int(
                    (gts[ind, 1] + gts[ind, 3]) / 2)
                dx= gaussian(x2 - x1)
                dy= gaussian(y2 - y1)
                gau_map= np.multiply(dy, np.transpose(dx))
                seman_map[y1:y2, x1:x2, 0]= np.maximum(
                    seman_map[y1:y2, x1:x2, 0], gau_map)
                seman_map[y1:y2, x1:x2, 1]= 1
                seman_map[c_y, c_x, 2]= 1

                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r +
                              1, 0]= np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1]= 1
                elif scale == 'w':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r +
                              1, 0]= np.log(gts[ind, 2] - gts[ind, 0])
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1]= 1
                elif scale == 'hw':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r +
                              1, 0]= np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r +
                              1, 1]= np.log(gts[ind, 2] - gts[ind, 0])
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 2]= 1
                if offset:
                    offset_map[c_y, c_x, 0]= (gts[ind, 1] +
                                               gts[ind, 3]) / 2 - c_y - 0.5
                    offset_map[c_y, c_x, 1]= (gts[ind, 0] +
                                               gts[ind, 2]) / 2 - c_x - 0.5
                    offset_map[c_y, c_x, 2]= 1
        return_data=[seman_map,scale_map]
        if offset:
            return_data.append(offset_map)
        if mask:
            return_data.append(mask_map)
        return return_data
