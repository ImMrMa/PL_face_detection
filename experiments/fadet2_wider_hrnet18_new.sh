cd src
# train
CUDA_VISIBLE_DEVICES=0,1 python main.py fadet2 --exp_id wider_hrnet2_why_norm --nms --arch hrnetv2_18 --batch_size 6 --wh_weight=1 --not_reg_offset --lr_dc 0.5 --lr 5e-4 --lr_step 15,35,50,60,65,65 --gpus 0,1  --num_workers 4 --resume --dataset wider

# test
# python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume
# # flip test
# python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test 
# # multi scale test
# python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
