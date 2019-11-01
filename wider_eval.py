import os
import time
import pickle
from tqdm import tqdm
import os.path as osp
import cv2
import numpy as np
import torch
from src.lib.models.networks.layer_conf import resnet18 as net
import torch.nn as nn
# import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '6'
cache_path = 'data/cache/widerface/val'
out_path = 'data/eval/resnet18_conv3_ori'
pretrained_path = '/data/users/mayx/my_code/github/CenterNet/models/model_conv3_ori.pth'
mode=0
def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel),
                                    stride=1,
                                    padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def get_model(pretrained_path=None):
    layer_conf = dict(
            layer2=dict(conv=dict(k=3, p=1, d=1, s=2),
                        down=dict(k=1, p=0, d=1, s=2)),
            layer3=dict(conv=dict(k=3, p=1, d=1, s=2),
                        down=dict(k=1, p=0, d=1, s=2)),
            layer4=dict(conv=dict(k=3, p=2, d=2, s=1),
                        down=dict(k=1, p=0, d=1, s=1)))
    model = net(layer_conf=layer_conf)
    if pretrained_path:
        print('loading weight!')
        model_dict=torch.load(pretrained_path, map_location='cpu')['state_dict']
        for k,v in model_dict.items():
            print(k,v.shape)
        input('s')
        model.load_state_dict(model_dict)
    return model


def parse_wider_offset(Y, img_h_new, img_w_new, score=0.1, down=4,
                       nmsthre=0.5):
    seman = Y[0][0, :, :, 0]
    height = Y[1][0, :, :, 0]
    width = Y[1][0, :, :, 1]
    offset_y = Y[2][0, :, :, 0]
    offset_x = Y[2][0, :, :, 1]
    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            w = np.exp(width[y_c[i], x_c[i]]) * down
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(
                0, (y_c[i] + o_y + 0.5) * down - h / 2)
            x1, y1 = min(x1, img_w_new), min(y1, img_h_new)
            boxs.append(
                [x1, y1,
                 min(x1 + w, img_w_new),
                 min(y1 + h, img_h_new), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        # keep = nms(boxs, nmsthre, usegpu=False, gpu_id=0)
        # boxs = boxs[keep, :]
        boxs = soft_bbox_vote(boxs, thre=nmsthre)
    return boxs


def soft_bbox_vote(det, thre=0.35, score=0.05):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= thre)[0]
        det_accu = det[merge_index, :]
        det_accu_iou = o[merge_index]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            soft_det_accu = det_accu.copy()
            soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
            soft_index = np.where(soft_det_accu[:, 4] >= score)[0]
            soft_det_accu = soft_det_accu[soft_index, :]

            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(
                det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(
                det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score

            if soft_det_accu.shape[0] > 0:
                det_accu_sum = np.row_stack((soft_det_accu, det_accu_sum))

            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    order = dets[:, 4].ravel().argsort()[::-1]
    dets = dets[order, :]
    return dets


mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3).astype(np.float32)
std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3).astype(np.float32)

with open(cache_path, 'rb') as fid:
    val_data = pickle.load(fid, encoding='latin1')
num_imgs = len(val_data)
print('num of val samples: {}'.format(num_imgs))
model = get_model(pretrained_path)
model.cuda()
model.eval()
if not os.path.exists(out_path):
    os.makedirs(out_path)
# files = sorted(os.listdir(w_path))
# get the results from epoch 51 to epoch 150
res_path = out_path
if not os.path.exists(res_path):
    os.makedirs(res_path)
print(res_path)

start_time = time.time()

for f in tqdm(range(num_imgs)):
    filepath = val_data[f]['filepath']
    event = filepath.split('/')[-2]
    event_path = osp.join(res_path, event)
    if not osp.exists(event_path):
        os.mkdir(event_path)
    filename = filepath.split('/')[-1].split('.')[0]
    txtpath = os.path.join(event_path, filename + '.txt')
    if os.path.exists(txtpath):
        continue
    img = cv2.imread(filepath)

    def pre_process(img):

        img = img.astype(np.float32)
        img = img / 255
        img = img[..., [2, 1, 0]]
        img = (img - mean) / std
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
        return img

    def detect_face(img, scale=1, flip=False,nms=True):
        img_h, img_w = img.shape[:2]
        img_h_new, img_w_new = int(np.ceil(scale * img_h / 16) * 16), int(
            np.ceil(scale * img_w / 16) * 16)
        scale_h, scale_w = img_h_new / img_h, img_w_new / img_w

        img_s = cv2.resize(img,
                           None,
                           None,
                           fx=scale_w,
                           fy=scale_h,
                           interpolation=cv2.INTER_LINEAR)
        # img_h, img_w = img_s.shape[:2]
        # print frame_number

        if flip:
            img_ = cv2.flip(img_s, 1)
            # x_rcnn = format_img_pad(img_sf, C)
            img = pre_process(img_)
        else:
            # x_rcnn = format_img_pad(img_s, C)
            img = pre_process(img_s)
        with torch.no_grad():
            output = model(img.cuda())
            if nms:
                output['hm'] = _nms(output['hm'])
            output = [
                output['hm'].cpu().detach().permute(0, 2, 3, 1).numpy(),
                output['wh'].cpu().detach().permute(0, 2, 3, 1).numpy(),
                output['offset'].cpu().detach().permute(0, 2, 3, 1).numpy()
            ]
        boxes = parse_wider_offset(output,
                                   img_h_new,
                                   img_w_new,
                                   score=0.05,
                                   nmsthre=0.6)
        if len(boxes) > 0:
            keep_index = np.where(
                np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] -
                           boxes[:, 1]) >= 12)[0]
            boxes = boxes[keep_index, :]
        if len(boxes) > 0:
            if flip:
                boxes[:, [0, 2]] = img_s.shape[1] - boxes[:, [2, 0]]
            boxes[:, 0:4:2] = boxes[:, 0:4:2] / scale_w
            boxes[:, 1:4:2] = boxes[:, 1:4:2] / scale_h
        else:
            boxes = np.empty(shape=[0, 5], dtype=np.float32)
        return boxes

    def detect_face_small_normal(img, scale=1, flip=False, nms=False):
        img_h, img_w = img.shape[:2]
        img_h_new, img_w_new = int(np.ceil(scale * img_h / 16) * 16), int(
            np.ceil(scale * img_w / 16) * 16)
        scale_h, scale_w = img_h_new / img_h, img_w_new / img_w

        img_s = cv2.resize(img,
                           None,
                           None,
                           fx=scale_w,
                           fy=scale_h,
                           interpolation=cv2.INTER_LINEAR)
        # img_h, img_w = img_s.shape[:2]
        # print frame_number

        if flip:
            img_ = cv2.flip(img_s, 1)
            # x_rcnn = format_img_pad(img_sf, C)
            img = pre_process(img_)
        else:
            # x_rcnn = format_img_pad(img_s, C)
            img = pre_process(img_s)
        with torch.no_grad():
            output = model(img.cuda())
            if nms:
                output['hm'] = _nms(output['hm'])
            if nms:
                output['hm_small'] = _nms(output['hm_small'])
            output_s = [
                output['hm_small'].cpu().detach().permute(0, 2, 3, 1).numpy(),
                output['wh_small'].cpu().detach().permute(0, 2, 3, 1).numpy(),
                output['offset_small'].cpu().detach().permute(0, 2, 3,
                                                              1).numpy()
            ]
            output_n = [
                output['hm'].cpu().detach().permute(0, 2, 3, 1).numpy(),
                output['wh'].cpu().detach().permute(0, 2, 3, 1).numpy(),
                output['offset'].cpu().detach().permute(0, 2, 3, 1).numpy()
            ]

        boxes_s = parse_wider_offset(output_s,
                                     img_h_new,
                                     img_w_new,
                                     score=0.05,
                                     nmsthre=0.6,
                                     down=2)
        boxes_n = parse_wider_offset(output_n,
                                     img_h_new,
                                     img_w_new,
                                     score=0.05,
                                     nmsthre=0.6)
        if len(boxes_s) > 0:
            keep_index = np.where(
                ((boxes_s[:, 2] - boxes_s[:, 0]) *
                 (boxes_s[:, 3] - boxes_s[:, 1]))**0.5 <= 14)[0]
            boxes_s = boxes_s[keep_index, :]
        if len(boxes_n) > 0:
            keep_index = np.where(
                ((boxes_n[:, 2] - boxes_n[:, 0]) *
                 (boxes_n[:, 3] - boxes_n[:, 1]))**0.5 >= 12)[0]
            boxes_n = boxes_n[keep_index, :]
        if len(boxes_s) > 0 and len(boxes_n) > 0:
            boxes = np.row_stack((boxes_s, boxes_n))
        elif len(boxes_s) > 0:
            boxes = boxes_s
        elif len(boxes_n) > 0:
            boxes = boxes_n
        else:
            boxes = np.empty(shape=[0, 5], dtype=np.float32)
        if len(boxes) > 0:
            if flip:
                boxes[:, [0, 2]] = img_s.shape[1] - boxes[:, [2, 0]]
            boxes[:, 0:4:2] = boxes[:, 0:4:2] / scale_w
            boxes[:, 1:4:2] = boxes[:, 1:4:2] / scale_h
        else:
            boxes = np.empty(shape=[0, 5], dtype=np.float32)
        return boxes

    def im_det_ms_pyramid(image, max_im_shrink, mode=0):
        if mode == 0:
            detector = detect_face
        elif mode == 1:
            detector = detect_face_small_normal
        # shrink detecting and shrink only detect big face
        det_s = np.row_stack((detector(image,
                                       0.5), detector(image, 0.5, flip=True)))
        index = np.where(
            np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] -
                       det_s[:, 1] + 1) > 64)[0]
        det_s = det_s[index, :]

        det_temp = np.row_stack(
            (detector(image, 0.75), detector(image, 0.75, flip=True)))
        index = np.where(
            np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] -
                       det_temp[:, 1] + 1) > 32)[0]
        det_temp = det_temp[index, :]
        det_s = np.row_stack((det_s, det_temp))

        det_temp = np.row_stack(
            (detector(image, 0.25), detector(image, 0.25, flip=True)))
        index = np.where(
            np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] -
                       det_temp[:, 1] + 1) > 96)[0]
        det_temp = det_temp[index, :]
        det_s = np.row_stack((det_s, det_temp))

        # st = [0.5, 1.5, 2.0]
        st = [1.25, 1.5, 1.75, 2.0, 2.25]
        for i in range(len(st)):
            if (st[i] <= max_im_shrink):
                det_temp = np.row_stack(
                    (detector(image, st[i]), detector(image, st[i],
                                                      flip=True)))
                # Enlarged images are only used to detect small faces.
                if st[i] == 1.25:
                    index = np.where(
                        np.minimum(det_temp[:, 2] - det_temp[:, 0] +
                                   1, det_temp[:, 3] - det_temp[:, 1] +
                                   1) < 128)[0]
                    det_temp = det_temp[index, :]
                elif st[i] == 1.5:
                    index = np.where(
                        np.minimum(det_temp[:, 2] - det_temp[:, 0] +
                                   1, det_temp[:, 3] - det_temp[:, 1] +
                                   1) < 96)[0]
                    det_temp = det_temp[index, :]
                elif st[i] == 1.75:
                    index = np.where(
                        np.minimum(det_temp[:, 2] - det_temp[:, 0] +
                                   1, det_temp[:, 3] - det_temp[:, 1] +
                                   1) < 64)[0]
                    det_temp = det_temp[index, :]
                elif st[i] == 2.0:
                    index = np.where(
                        np.minimum(det_temp[:, 2] - det_temp[:, 0] +
                                   1, det_temp[:, 3] - det_temp[:, 1] +
                                   1) < 48)[0]
                    det_temp = det_temp[index, :]
                elif st[i] == 2.25:
                    index = np.where(
                        np.minimum(det_temp[:, 2] - det_temp[:, 0] +
                                   1, det_temp[:, 3] - det_temp[:, 1] +
                                   1) < 32)[0]
                    det_temp = det_temp[index, :]
                elif st[i] == 0.25:
                    index = np.where(
                        np.minimum(det_temp[:, 2] - det_temp[:, 0] +
                                   1, det_temp[:, 3] - det_temp[:, 1] +
                                   1) < 64)[0]
                    det_temp = det_temp[index, :]
                elif st[i] == 0.5:
                    index = np.where(
                        np.minimum(det_temp[:, 2] - det_temp[:, 0] +
                                   1, det_temp[:, 3] - det_temp[:, 1] +
                                   1) < 128)[0]
                    det_temp = det_temp[index, :]
                det_s = np.row_stack((det_s, det_temp))
        return det_s

    max_im_shrink = (
        0x7fffffff / 577.0 /
        (img.shape[0] * img.shape[1]))**0.5  # the max size of input image
    shrink = max_im_shrink if max_im_shrink < 1 else 1
    if mode==1:
        detector=detect_face_small_normal
    else:
        detector=detect_face
    det0 = detector(
        img,
        nms=True,
    )
    # det1 = detector(img, flip=True,nms=False)
    # det2 = im_det_ms_pyramid(img, max_im_shrink,mode=mode)
    # merge all test results via bounding box voting
    # det = np.row_stack((det0, det1, det2))
    det=det0
    keep_index = np.where(np.minimum(det[:, 2] - det[:, 0], det[:, 3] - det[:, 1]) >= 3)[0]
    det = det[keep_index, :]
    dets = soft_bbox_vote(det, thre=0.4)
    # keep_index = np.where((dets[:, 2] - dets[:, 0] + 1) * (dets[:, 3] - dets[:, 1] + 1) >= 6 ** 2)[0]
    # dets = dets[keep_index, :]
    with open(txtpath, 'w') as f:
        f.write('{:s}\n'.format(filename))
        f.write('{:d}\n'.format(len(dets)))
        for line in dets:
            f.write('{:.0f} {:.0f} {:.0f} {:.0f} {:.3f}\n'.format(
                line[0], line[1], line[2] - line[0] + 1, line[3] - line[1] + 1,
                line[4]))
print(time.time() - start_time)