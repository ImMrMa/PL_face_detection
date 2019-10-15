cd src
# train
CUDA_VISIBLE_DEVICES=0 python test.py fadet2 --exp_id wider_hrnet2_eval --arch hrnetv2_18 --down_ratio 1  --test_scales 0.5,0.75,1,1.25,1.5 --nms --not_reg_offset   --num_workers 4  --dataset wider --resume --keep_res

# test
# python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume
# # flip test
# python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test 
# # multi scale test
# python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
