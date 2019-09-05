from datasets.multi_res_dataset.datasets.dataset_factory import get_dataset
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import random


def multi_data():
    gen_dataset = get_dataset('pascal', 'ctdet')

    class Opts():
        data_dir='../data'
        keep_res=False
        input_h=512
        input_w=512
        down_ratio=4
        mse_loss=False
        dense_wh=False
        cat_spec_wh=False
        reg_offset=True
        debug=0
        draw_ma_gaussian=False
        not_rand_crop=True
        scale=0
        shift=0
        flip=0
        no_color_aug=True
    opt = Opts()
    dataset_pascal = gen_dataset(opt, 'train')
    objs = torch.load('../models/objs_2')

    class PascalDataset(Dataset):
        def __init__(self,dataset,obj_res,pic_res,objs):
            self.objs=objs['objs_'+str(obj_res)]['pics_'+str(pic_res)]
            self.dataset=dataset
            self.obj_res=obj_res
            self.pic_res=pic_res
            self.avgpool=torch.nn.MaxPool2d(4,4)
            self.maxpool=torch.nn.MaxPool2d(4,4)
            self.max_objs=50
        def __len__(self):
            return len(self.objs)
        def __getitem__(self, index):
            dataset=self.dataset
            objs=self.objs
            pic_index=objs[index]['pic_index']
            obj_index=objs[index]['obj_index']
            obj_pic=dataset[pic_index]
            
            wh=obj_pic['wh'][obj_index]
            ori_pic=obj_pic['input']
            bbox=(obj_pic['bboxs'][obj_index]).astype(np.int)
            hms=obj_pic['hm']
            bbox_crop=self.crop_pic(ori_pic,bbox,wh)
            batch_dict=self.gen_offset(obj_pic['bboxs'],bbox_crop)
            crop_pic=ori_pic[:,bbox_crop[1]:bbox_crop[3],bbox_crop[0]:bbox_crop[2]]
            crop_hm=hms[:,bbox_crop[1]:bbox_crop[3],bbox_crop[0]:bbox_crop[2]]
            crop_resize_hm=self.avgpool(torch.Tensor(crop_hm))
            batch_dict['hm']=crop_resize_hm
            batch_dict['input']=crop_pic
            return batch_dict
        def gen_offset(self,obj_bboxes,bbox_crop):
            reg = np.zeros((self.max_objs, 2), dtype=np.float32)
            ind = np.zeros((self.max_objs), dtype=np.int64)
            reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
            wh=np.zeros((self.max_objs,2),dtype=np.float32)
            
            bbox_output=bbox_crop/4
            w=bbox_output[2]-bbox_output[0]
            h=bbox_output[3]-bbox_output[1]

            for index,obj_bbox in enumerate(obj_bboxes):
                if obj_bbox[0]>=bbox_crop[0] and obj_bbox[1]>=bbox_crop[1] and obj_bbox[2]<=bbox_crop[2] and obj_bbox[3]<=bbox_crop[3]:
                    obj_bbox_offset=obj_bbox-[bbox_crop[0],bbox_crop[1],bbox_crop[0],bbox_crop[1]]
                    ct = np.array([(obj_bbox_offset[0] + obj_bbox_offset[2]) / 2, (obj_bbox_offset[1] + obj_bbox_offset[3]) / 2], dtype=np.float32)
                    ct=ct/4
                    ct_int =ct.astype(np.int32)
                    reg[index]=ct-ct_int
                    index_ct=ct_int[1]*w+ct_int[0]
                    wh[index]=[(obj_bbox_offset[2]-obj_bbox_offset[0])/4,(obj_bbox_offset[3]-obj_bbox_offset[1])/4]
                    if index<0 or index>=w*h:
                        reg_mask[index]=0
                        ind[index]=0
                    else:
                        reg_mask[index]=1
                        ind[index]=index_ct
            wh_and_offset=dict(reg=reg,ind=ind,reg_mask=reg_mask,wh=wh)
            return wh_and_offset
                    
        def crop_pic(self,pic,bbox,wh):
            bbox_crop=np.zeros(4,np.int)
            ori_w=pic.shape[2]
            ori_h=pic.shape[1]
            cut_w,cut_h=self.pic_res,self.pic_res
            if bbox[1]<ori_h-bbox[3]:
                max_h,h_l=bbox[1],True
                min_h=(cut_h+bbox[1])-ori_h
            else:
                max_h,h_l=(ori_h-bbox[3]),False
                min_h=cut_h-bbox[3]
            if bbox[0]<ori_w-bbox[2]:
                max_w,w_l=bbox[0],True
                min_w=(cut_w+bbox[0])-ori_w
            else:
                max_w,w_l=(ori_w-bbox[2]),False
                min_w=cut_w-bbox[2]
            max_h=min(max_h,cut_h-wh[1])
            max_w=min(max_w,cut_w-wh[0])
            min_h=max(0,min_h)
            min_w=max(0,min_w)
            
            rand_h=np.random.randint(min_h,max_h+1)
            rand_w=np.random.randint(min_w,max_w+1)

            if h_l:
                bbox_crop[1]=bbox[1]-rand_h
                bbox_crop[3]=bbox_crop[1]+cut_h
            else:
                bbox_crop[3]=bbox[3]+rand_h
                bbox_crop[1]=bbox_crop[3]-cut_h
            if w_l:
                bbox_crop[0]=bbox[0]-rand_w
                bbox_crop[2]=bbox_crop[0]+cut_w
            else:
                bbox_crop[2]=bbox[2]+rand_w
                bbox_crop[0]=bbox_crop[2]-cut_w

            return bbox_crop



    class DatasetObj(Dataset):
        def __init__(self, dataset_obj, dataset, objs, obj_res, pic_res, loader_bses):
            dataloaders = []
            dataloader_size = []
            for res, loader_bs in zip(pic_res, loader_bses):
                dataloaders.append(iter(DataLoader(dataset_obj(
                    dataset, obj_res, res, objs), batch_size=loader_bs, shuffle=True)))
            sum_len = 0
            for dataloader in dataloaders:
                sum_len += len(dataloader)
                dataloader_size.append(len(dataloader))
            self.dataloaders = dataloaders
            self.dataloader_size = dataloader_size
            self.sum_len = sum_len
            self.random_pool = self.dataloader_size.copy()

        def __getitem__(self, index):
            indexs = [i for i in range(
                len(self.random_pool)) if self.random_pool[i] > 0]
            loader_index = random.choice(indexs)

            self.random_pool[loader_index] -= 1
            batch = self.dataloaders[loader_index].next()
            if index == self.sum_len-1:
                self.random_pool = self.dataloader_size.copy()
            return batch

        def __len__(self):
            return self.sum_len

    class DatasetObjMuiltRes(Dataset):
        def __init__(self, objs, dataset, obj_res=[], pic_res=dict(), loader_bses=dict()):
            self.obj_res = obj_res
            dataloaders = []

            def default_collate(batch):
                return batch[0]
            for res in obj_res:
                dataset_obj = DatasetObj(
                    PascalDataset, dataset, objs, res, pic_res['pic_'+str(res)], loader_bses['pic_'+str(res)])
                dataloaders.append(
                    iter(DataLoader(dataset_obj, num_workers=0, collate_fn=default_collate)))
            dataloader_size = []
            sum_len = 0
            for dataloader in dataloaders:
                sum_len += len(dataloader)
                dataloader_size.append(len(dataloader))
            self.dataloaders = dataloaders
            self.dataloader_size = dataloader_size
            self.sum_len = sum_len
            self.random_pool = self.dataloader_size.copy()

        def __getitem__(self, index):
            indexs = [i for i in range(
                len(self.random_pool)) if self.random_pool[i] > 0]
            loader_index = random.choice(indexs)
            self.random_pool[loader_index] -= 1
            batch = self.dataloaders[loader_index].next()
            if index == self.sum_len-1:
                self.random_pool = self.dataloader_size.copy()
            batch['res'] = self.obj_res[loader_index]
            return batch

        def __len__(self):
            return self.sum_len

    # data = DatasetObjMuiltRes(objs,
    #                           dataset_pascal,
    #                           obj_res=[32, 64, 128, 256],
    #                           pic_res=dict(pic_32=[64, 128, 192, 256],
    #                                        pic_64=[128, 192, 256, 384],
    #                                        pic_128=[256, 384, 512],
    #                                        pic_256=[384, 512]),
    #                           loader_bses=dict(pic_32=[256, 64, 32, 16],
    #                                            pic_64=[64, 32, 16, 12],
    #                                            pic_128=[16, 12, 8], 
    #                                            pic_256=[12, 8]))
    mother_batch=8
    data = DatasetObjMuiltRes(objs,
                              dataset_pascal,
                              obj_res=[32, 64, 128, 256],
                              pic_res=dict(pic_32=[64, 128, 192, 256],
                                           pic_64=[128, 192, 256, 384],
                                           pic_128=[256, 384, 512],
                                           pic_256=[384, 512]),
                              loader_bses=dict(pic_32=[mother_batch*16, mother_batch*8,mother_batch*6, mother_batch*4],
                                               pic_64=[mother_batch*8, mother_batch*6, 4*mother_batch,int(mother_batch*3/2)],
                                               pic_128=[4*mother_batch, int(mother_batch*3/2), mother_batch], 
                                               pic_256=[int(mother_batch*3/2), mother_batch]))
    def default_collate(batch):
        return batch[0]
    loader = DataLoader(data, num_workers=10, collate_fn=default_collate)
    return loader
