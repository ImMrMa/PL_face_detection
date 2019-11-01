import cv2
import numpy as np
import copy
import os.path as osp
def _brightness(image, min=0.5, max=2.0):
    '''
    Randomly change the brightness of the input image.

    Protected against overflow.
    '''
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    random_br = np.random.uniform(min,max)

    #To protect against overflow: Calculate a mask for all pixels
    #where adjustment of the brightness would exceed the maximum
    #brightness value and set the value to the maximum at those pixels.
    mask = hsv[:,:,2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:,:,2] * random_br)
    hsv[:,:,2] = v_channel

    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

def resize_image(image, gts,igs, scale=[0.4,1.5]):
    height, width = image.shape[0:2]
    ratio = np.random.uniform(scale[0], scale[1])
    # if len(gts)>0 and np.max(gts[:,3]-gts[:,1])>300:
    #     ratio = np.random.uniform(scale[0], 1.0)
    new_height, new_width = int(ratio*height), int(ratio*width)
    image = cv2.resize(image, (new_width, new_height))
    if len(gts)>0:
        gts = np.asarray(gts,dtype=float)
        gts[:, 0:4:2] *= ratio
        gts[:, 1:4:2] *= ratio

    if len(igs)>0:
        igs = np.asarray(igs, dtype=float)
        igs[:, 0:4:2] *= ratio
        igs[:, 1:4:2] *= ratio

    return image, gts, igs

def random_crop(image, gts, igs, crop_size, limit=8):
    img_height, img_width = image.shape[0:2]
    crop_h, crop_w = crop_size

    if len(gts)>0:
        sel_id = np.random.randint(0, len(gts))
        sel_center_x = int((gts[sel_id, 0] + gts[sel_id, 2]) / 2.0)
        sel_center_y = int((gts[sel_id, 1] + gts[sel_id, 3]) / 2.0)
    else:
        sel_center_x = int(np.random.randint(0, img_width - crop_w+1) + crop_w * 0.5)
        sel_center_y = int(np.random.randint(0, img_height - crop_h+1) + crop_h * 0.5)

    crop_x1 = max(sel_center_x - int(crop_w * 0.5), int(0))
    crop_y1 = max(sel_center_y - int(crop_h * 0.5), int(0))
    diff_x = max(crop_x1 + crop_w - img_width, int(0))
    crop_x1 -= diff_x
    diff_y = max(crop_y1 + crop_h - img_height, int(0))
    crop_y1 -= diff_y
    cropped_image = np.copy(image[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w])
    # crop detections
    if len(igs)>0:
        igs[:, 0:4:2] -= crop_x1
        igs[:, 1:4:2] -= crop_y1
        igs[:, 0:4:2] = np.clip(igs[:, 0:4:2], 0, crop_w)
        igs[:, 1:4:2] = np.clip(igs[:, 1:4:2], 0, crop_h)
        keep_inds = ((igs[:, 2] - igs[:, 0]) >=8) & \
                    ((igs[:, 3] - igs[:, 1]) >=8)
        igs = igs[keep_inds]
    if len(gts)>0:
        ori_gts = np.copy(gts)
        gts[:, 0:4:2] -= crop_x1
        gts[:, 1:4:2] -= crop_y1
        gts[:, 0:4:2] = np.clip(gts[:, 0:4:2], 0, crop_w)
        gts[:, 1:4:2] = np.clip(gts[:, 1:4:2], 0, crop_h)

        before_area = (ori_gts[:, 2] - ori_gts[:, 0]) * (ori_gts[:, 3] - ori_gts[:, 1])
        after_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])

        keep_inds = ((gts[:, 2] - gts[:, 0]) >=limit) & \
                    (after_area >= 0.5 * before_area)
        gts = gts[keep_inds]

    return cropped_image, gts, igs


def random_pave(image, gts, igs, pave_size, limit=8,mask=False):
    img_height, img_width = image.shape[0:2]
    pave_h, pave_w = pave_size
    # paved_image = np.zeros((pave_h, pave_w, 3), dtype=image.dtype)
    paved_image = np.ones((pave_h, pave_w, 3), dtype=image.dtype) * np.mean(image, dtype=int)
    pave_x = int(np.random.randint(0, pave_w - img_width + 1))
    pave_y = int(np.random.randint(0, pave_h - img_height + 1))
    paved_image[pave_y:pave_y + img_height, pave_x:pave_x + img_width] = image
    # pave detections
    if len(igs) > 0:
        igs[:, 0:4:2] += pave_x
        igs[:, 1:4:2] += pave_y
        keep_inds = ((igs[:, 2] - igs[:, 0]) >= 8) & \
                    ((igs[:, 3] - igs[:, 1]) >= 8)
        igs = igs[keep_inds]

    if len(gts) > 0:
        gts[:, 0:4:2] += pave_x
        gts[:, 1:4:2] += pave_y
        if not mask:
            keep_inds = ((gts[:, 2] - gts[:, 0]) >= limit)
            gts = gts[keep_inds]

    return paved_image, gts, igs

def augment(img_data, c):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    img_data_aug = copy.deepcopy(img_data)
    img = cv2.imread(os.path.join('../',img_data_aug['filepath']))
    print(os.path.join('../',img_data_aug['filepath']))
    input('s')
    img_height, img_width = img.shape[:2]

    # random brightness
    if c.brightness and np.random.randint(0, 2) == 0:
        img = _brightness(img, min=c.brightness[0], max=c.brightness[1])
    # random horizontal flip
    if c.use_horizontal_flips and np.random.randint(0, 2) == 0:
        img = cv2.flip(img, 1)
        if len(img_data_aug['bboxes']) > 0:
            img_data_aug['bboxes'][:, [0, 2]] = img_width - img_data_aug['bboxes'][:, [2, 0]]
        if len(img_data_aug['ignoreareas']) > 0:
            img_data_aug['ignoreareas'][:, [0, 2]] = img_width - img_data_aug['ignoreareas'][:, [2, 0]]

    gts = np.copy(img_data_aug['bboxes'])
    igs = np.copy(img_data_aug['ignoreareas'])

    img, gts, igs = resize_image(img, gts, igs, scale=[0.4,1.5])
    if img.shape[0]>=c.size_train[0]:
        img, gts, igs = random_crop(img, gts, igs, c.size_train,limit=16)
    else:
        img, gts, igs = random_pave(img, gts, igs, c.size_train,limit=16)

    img_data_aug['bboxes'] = gts
    img_data_aug['ignoreareas'] = igs

    img_data_aug['width'] = c.size_train[1]
    img_data_aug['height'] = c.size_train[0]

    return img_data_aug, img

def augment_wider(img_data, c,mask,resmaple=True,default_resample=False,limit=4):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    img_data_aug = copy.deepcopy(img_data)
    img = cv2.imread(osp.join('../',img_data_aug['filepath']))
    img_height, img_width = img.shape[:2]

    # random brightness
    if c.brightness and np.random.randint(0, 2) == 0:
        img = _brightness(img, min=c.brightness[0], max=c.brightness[1])
    # random horizontal flip
    if c.use_horizontal_flips and np.random.randint(0, 2) == 0:
        img = cv2.flip(img, 1)
        if len(img_data_aug['bboxes']) > 0:
            img_data_aug['bboxes'][:, [0, 2]] = img_width - img_data_aug['bboxes'][:, [2, 0]]
    gts = np.copy(img_data_aug['bboxes'])
    if default_resample:
        scales = np.asarray([16, 32, 64, 128, 256])
    else:
        scales = np.asarray([8,16, 32, 64, 128, 256])
    crop_p = c.size_train[0]
    dont_change=False
    if len(gts) > 0:
        sel_id = np.random.randint(0, len(gts))
        s_face = np.sqrt((gts[sel_id, 2] - gts[sel_id, 0]) * (gts[sel_id, 3] - gts[sel_id, 1]))
        if resmaple:
            last_index=np.argmin(np.abs(scales - s_face))
            if mask:
                last1=1
                if last_index<=3:
                    last1=2
                index = np.random.randint(max(0,last_index-1), min(last_index + last1,6))
                s_tar = np.random.uniform(np.power(2, 3 + index)*1, np.power(2, 3 + index) * 2)
            elif default_resample:
                index = np.random.randint(0, last_index+1)
                s_tar = np.random.uniform(np.power(2, 4 + index)*1.5, np.power(2, 4 + index) * 2)
            else:
                if last_index<=1:
                    index = np.random.randint(last_index, last_index+3)
                    s_tar = np.random.uniform(np.power(2, 3 + index)*1, np.power(2, 3 + index) * 2)
                else:
                    index=last_index
                    # index = np.random.randint(1, last_index+1)
                    # s_tar = np.random.uniform(np.power(2, 3 + index)*1.5, np.power(2, 3 + index) * 2)
                if index==last_index:
                    dont_change=True
        else:
            dont_change=True
        if not dont_change:
            ratio = s_tar / s_face
            new_height, new_width = int(ratio * img_height), int(ratio * img_width)
            img = cv2.resize(img, (new_width, new_height))
            gts = np.asarray(gts, dtype=float) * ratio
        else:
            ratio=1.0
            new_height, new_width=img_height,img_width
            gts = np.asarray(gts, dtype=float) * ratio
        crop_x1 = np.random.randint(0, int(gts[sel_id, 0])+1)
        crop_x1 = np.minimum(crop_x1, np.maximum(0, new_width - crop_p))
        crop_y1 = np.random.randint(0, int(gts[sel_id, 1])+1)
        crop_y1 = np.minimum(crop_y1, np.maximum(0, new_height - crop_p))
        img = img[crop_y1:crop_y1 + crop_p, crop_x1:crop_x1 + crop_p]
        # crop detections
        if len(gts) > 0:
            ori_gts = np.copy(gts)
            gts[:, 0:4:2] -= crop_x1
            gts[:, 1:4:2] -= crop_y1
            gts[:, 0:4:2] = np.clip(gts[:, 0:4:2], 0, crop_p)
            gts[:, 1:4:2] = np.clip(gts[:, 1:4:2], 0, crop_p)

            before_area = (ori_gts[:, 2] - ori_gts[:, 0]) * (ori_gts[:, 3] - ori_gts[:, 1])
            after_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])

            keep_inds = (after_area >= 0.5 * before_area)
            keep_inds_ig = (after_area < 0.5 * before_area)
            igs = gts[keep_inds_ig]
            gts = gts[keep_inds]

            if len(igs) > 0:
                w, h = igs[:, 2] - igs[:, 0], igs[:, 3] - igs[:, 1]
                igs = igs[np.logical_and(w >= limit, h >= limit), :]
            if len(gts) > 0:
                w, h = gts[:, 2] - gts[:, 0], gts[:, 3] - gts[:, 1]
                gts = gts[np.logical_and(w >= limit, h >= limit), :]
    else:
        img = img[0:crop_p, 0:crop_p]

    # scales = np.asarray([8,16, 32, 64, 128, 256])
    # crop_p = c.size_train[0]
    # if len(gts) > 0:
    #     sel_id = np.random.randint(0, len(gts))
    #     s_face = np.sqrt((gts[sel_id, 2] - gts[sel_id, 0]) * (gts[sel_id, 3] - gts[sel_id, 1]))
    #     last_index=np.argmin(np.abs(scales - s_face)) + 1
    #     last_list=np.array(range(max(0,last_index-2),last_index))
    #     if last_index>3:
    #         last_list=np.array([3,4,5])
    #         index=np.random.choice(last_list,p=[0.25,0.3,0.45])   
    #     else:
    #         index = np.random.choice(last_list)
    #     s_tar = np.random.uniform(np.power(2, 3 + index), np.power(2, 3 + index) * 2)
    #     ratio = round(s_tar / s_face,4)
    #     try:
    #         new_height, new_width = int(ratio * img_height), int(ratio * img_width)
    #     except:
    #         print(ratio,s_face,gts[sel_id],img_height,img_width)    
    #         input('s')
    #     img = cv2.resize(img, (new_width, new_height))
    #     gts = np.asarray(gts, dtype=float) * ratio

    #     crop_x1 = np.random.randint(0, int(gts[sel_id, 0])+1)
    #     crop_x1 = np.minimum(crop_x1, np.maximum(0, new_width - crop_p))
    #     crop_y1 = np.random.randint(0, int(gts[sel_id, 1])+1)
    #     crop_y1 = np.minimum(crop_y1, np.maximum(0, new_height - crop_p))
    #     img = img[crop_y1:crop_y1 + crop_p, crop_x1:crop_x1 + crop_p]
    #     # crop detections
    #     if len(gts) > 0:
    #         ori_gts = np.copy(gts)
    #         gts[:, 0:4:2] -= crop_x1
    #         gts[:, 1:4:2] -= crop_y1
    #         gts[:, 0:4:2] = np.clip(gts[:, 0:4:2], 0, crop_p)
    #         gts[:, 1:4:2] = np.clip(gts[:, 1:4:2], 0, crop_p)

    #         before_area = (ori_gts[:, 2] - ori_gts[:, 0]) * (ori_gts[:, 3] - ori_gts[:, 1])
    #         after_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])

    #         keep_inds = (after_area >= 0.5 * before_area)
    #         keep_inds_ig = (after_area < 0.5 * before_area)
    #         igs = gts[keep_inds_ig]
    #         gts = gts[keep_inds]

    #         if len(igs) > 0:
    #             w, h = igs[:, 2] - igs[:, 0], igs[:, 3] - igs[:, 1]
    #             igs = igs[np.logical_and(w >= 5, h >= 5), :]
    #         if len(gts) > 0:
    #             w, h = gts[:, 2] - gts[:, 0], gts[:, 3] - gts[:, 1]
    #             gts = gts[np.logical_and(w >= 5, h >= 5), :]
    # else:
    #     img = img[0:crop_p, 0:crop_p]

    if np.minimum(img.shape[0], img.shape[1]) < c.size_train[0]:
        img, gts, igs = random_pave(img, gts, igs, c.size_train,mask=mask,limit=limit)

    img_data_aug['bboxes'] = gts
    img_data_aug['ignoreareas'] = igs
    return img_data_aug, img
