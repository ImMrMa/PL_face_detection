#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import math
from PIL import Image, ImageDraw
import torch.utils.data as data
import numpy as np
import random
from .augmentations import preprocess
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian,gen_quard
from utils.config import cfg
class WIDERDetection(data.Dataset):
    """docstring for WIDERDetection"""
    default_resolution=cfg.resize_height,cfg.resize_width
    mean = np.array([0.485, 0.456, 0.406],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)
    num_classes=1
    def __init__(self,opts=None, mode='train'):
        if mode=='train':
            list_file=cfg.FACE.TRAIN_FILE
        else:
            list_file=cfg.FACE.VAL_FILE
        super(WIDERDetection, self).__init__()
        self.mode = mode
        self.fnames = []
        self.boxes = []
        self.labels = []
        
        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            num_faces = int(line[1])
            box = []
            label = []
            for i in range(num_faces):
                x = float(line[2 + 5 * i])
                y = float(line[3 + 5 * i])
                w = float(line[4 + 5 * i])
                h = float(line[5 + 5 * i])
                c = int(line[6 + 5 * i])
                if w <= 0 or h <= 0:
                    continue
                box.append([x, y, x + w, y + h])
                label.append(c)
            if len(box) > 0:
                self.fnames.append(line[0])
                self.boxes.append(box)
                self.labels.append(label)

        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, target, h, w = self.pull_item(index)
        ret=self.transform(img,target)
        return ret
    def gen_labels(self,h,w,bbox_index,hm,wh,offset,mask,stride):
        rf_sizes=[[10,32],[32,64],[64,128],[128,384]]
        index=stride-1
        wh=wh[index]
        mask=mask[index]
        draw_gaussian =draw_umich_gaussian
        if (w+h)<=16:
            radius=math.sqrt(((math.ceil(h)*math.ceil(w))/6.28))
        else:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, math.ceil(radius))
        ct=np.array([(bbox_index[0]+bbox_index[2])/2,(bbox_index[1]+bbox_index[3])/2])
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[1],ct_int,radius)

        wh[0][ct_int[1],ct_int[0]]=0.6*(w-rf_sizes[index][0])/(rf_sizes[index][1]-rf_sizes[index][0])+0.2
        wh[1][ct_int[1],ct_int[0]]=0.6*(h-rf_sizes[index][0])/(rf_sizes[index][1]-rf_sizes[index][0])+0.2
        offset[0][ct_int[1],ct_int[0]]=ct[0]-ct_int[0]
        offset[1][ct_int[1],ct_int[0]]=ct[1]-ct_int[1]
        mask[ct_int[1],ct_int[0]]=1
    def gen_labels_two(self,h,w,bbox_index,hm_small,hm_norm,wh_small,wh_norm,mask_small,mask_norm):
        draw_gaussian =draw_umich_gaussian
        length=(h*w)**0.5
        if length<=64:
            mode=1
        else:
            mode=2
        if length<=16:
            radius=length
        elif length<=32: 
            radius =math.ceil(length*(1-0.3*(length-16)/16))
        elif length<=64:
            radius=(length*0.7)*(1-(length-32)/32)+gaussian_radius((math.ceil(h), math.ceil(w)))*((length-32)/32)
        else:
            radius=gaussian_radius((math.ceil(h), math.ceil(w)))
        
        radius = max(1, math.ceil(radius))
        ct=np.array([(bbox_index[0]+bbox_index[2])/2,(bbox_index[1]+bbox_index[3])/2])
        ct_int = ct.astype(np.int32)

        if length<=192:
            draw_gaussian(hm_small[0],ct_int,radius)
        if (h+w)/2<=32:
            rf_size=32
            norm_w=w/32*0.3
            norm_h=h/32*0.3
        elif (h+w)/2<=128:
            rf_sizes=[32,128]
            norm_w=0.4*(w-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.3
            norm_h=0.4*(h-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.3
        elif (h+w)/2<=384:
            rf_sizes=[128,384]
            norm_w=0.2*(w-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.7
            norm_h=0.2*(h-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.7
        else:
            rf_sizes=[384,1024]
            norm_w=0.08*(w-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.9
            norm_h=0.08*(h-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.9
        wh_small[0][ct_int[1],ct_int[0]]=norm_w
        wh_small[1][ct_int[1],ct_int[0]]=norm_h   
        mask_small[0][ct_int[1],ct_int[0]]=1

        if mode==2:
            h,w=h,w
            bbox_index=[bbox_index[0]/4,bbox_index[1]/4,bbox_index[2]/4,bbox_index[3]/4]
            radius=gaussian_radius((math.ceil(h/4), math.ceil(w/4)))
            radius = max(1, math.ceil(radius))  
            ct=np.array([(bbox_index[0]+bbox_index[2])/2,(bbox_index[1]+bbox_index[3])/2])
            ct_int = ct.astype(np.int32)
            draw_gaussian(hm_norm[0],ct_int,radius)
            if (h+w)/2<=128:
                rf_sizes=[64,128]
                norm_w=0.2*(w-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.1
                norm_h=0.2*(h-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.1
            elif (h+w)/2<=640:
                rf_sizes=[128,768]
                norm_w=0.5*(w-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.3
                norm_h=0.5*(h-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.3    
            else:
                rf_sizes=[768,1024]
                norm_w=0.1*(w-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.8
                norm_h=0.1*(h-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.8
            wh_norm[0][ct_int[1],ct_int[0]]=norm_w
            wh_norm[1][ct_int[1],ct_int[0]]=norm_h   
            mask_norm[0][ct_int[1],ct_int[0]]=1
    def gen_labels_single(self,h,w,bbox_index,hm,wh,offset,mask):
        draw_gaussian =draw_umich_gaussian
        radius=math.sqrt(((math.ceil(h)*math.ceil(w))/6.28))
        # radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, math.ceil(radius))

        ct=np.array([(bbox_index[0]+bbox_index[2])/2,(bbox_index[1]+bbox_index[3])/2])
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[1],ct_int,radius)
        if (h+w)/2<=32:
            rf_size=32
            norm_w=w/32*0.3
            norm_h=h/32*0.3
        elif (h+w)/2<=128:
            rf_sizes=[32,128]
            norm_w=0.4*(w-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.3
            norm_h=0.4*(h-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.3
        elif (h+w)/2<=384:
            rf_sizes=[128,384]
            norm_w=0.2*(w-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.7
            norm_h=0.2*(h-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.7
        else:
            rf_sizes=[384,1024]
            norm_w=0.08*(w-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.9
            norm_h=0.08*(h-rf_sizes[0])/(rf_sizes[1]-rf_sizes[0])+0.9
        wh[0][ct_int[1],ct_int[0]]=norm_w
        wh[1][ct_int[1],ct_int[0]]=norm_h   
        offset[0][ct_int[1],ct_int[0]]=ct[0]-ct_int[0]
        offset[1][ct_int[1],ct_int[0]]=ct[1]-ct_int[1]
        mask[0][ct_int[1],ct_int[0]]=1
    def transform(self,img,target):

        output_h=img.shape[1]
        output_w=img.shape[2]
        if cfg.multi_wh:
            wh_small=np.zeros((2, output_h, output_w), dtype=np.float32)
            wh_norm=np.zeros((2, output_h//4, output_w//4), dtype=np.float32)
            mask_small=np.zeros((1,output_h, output_w),dtype=np.float32)
            mask_norm=np.zeros((1,output_h//4,output_w//4),dtype=np.float32)
            hm_small=np.zeros((1,output_h,output_w),dtype=np.float32)
            hm_norm=np.zeros((1,output_h//4,output_w//4),dtype=np.float32)
        else:
            wh=np.zeros((2, output_h, output_w), dtype=np.float32)
            mask=np.zeros((1,output_h, output_w),dtype=np.float32)
        for bbox in target:
            bbox_index=[bbox[0]*output_w,bbox[1]*output_h,bbox[2]*output_w,bbox[3]*output_h]
            w,h=(bbox_index[2]-bbox_index[0]),(bbox_index[3]-bbox_index[1])
            if cfg.multi_wh:
                self.gen_labels_two(h,w,bbox_index,hm_small,hm_norm,wh_small,wh_norm,mask_small,mask_norm)
                # if (w+h)/2<=32:
                #     self.gen_labels(h,w,bbox_index,hm,wh,offset,mask,1)
                # elif (w+h)/2<=64:
                #     self.gen_labels(h,w,bbox_index,hm,wh,offset,mask,2)
                # elif (w+h)/2<=128:
                #     self.gen_labels(h,w,bbox_index,hm,wh,offset,mask,3)
                # else:
                #     self.gen_labels(h,w,bbox_index,hm,wh,offset,mask,4)
            else:
                self.gen_labels_single(h,w,bbox_index,hm,wh,offset,mask)
        if cfg.multi_wh:
            ret = {'input': img, 'hm_small': hm_small,'wh_small': wh_small,'mask_small': mask_small,'mask_norm':mask_norm,'wh_norm':wh_norm,'hm_norm':hm_norm}
        else:
            ret= {'input': img, 'hm': hm,'wh': wh,
            'mask': mask, }
        return ret
    def pull_item(self, index):
        while True:
            image_path = self.fnames[index]
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')
    
            im_width, im_height = img.size
            boxes = self.annotransform(
                np.array(self.boxes[index]), im_width, im_height)
            label = np.array(self.labels[index])
            bbox_labels = np.hstack((label[:, np.newaxis], boxes)).tolist()
            img, sample_labels = preprocess(
                img, bbox_labels, self.mode, image_path)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) > 0:
                target = np.hstack(
                    (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))

                assert (target[:, 2] > target[:, 0]).any()
                assert (target[:, 3] > target[:, 1]).any()
                break 
            else:
                index = random.randrange(0, self.num_samples)

        
        #img = Image.fromarray(img)
        '''
        draw = ImageDraw.Draw(img)
        w,h = img.size
        for bbox in sample_labels:
            bbox = (bbox[1:] * np.array([w, h, w, h])).tolist()

            draw.rectangle(bbox,outline='red')
        img.save('image.jpg')
        '''
        return torch.from_numpy(img), target, im_height, im_width
        

    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


if __name__ == '__main__':
    from config import cfg
    dataset = WIDERDetection(cfg.FACE.TRAIN_FILE)
    #for i in range(len(dataset)):
    dataset.pull_item(14)
