cd src
# train
CUDA_VISIBLE_DEVICES=0 python main.py fadet --exp_id wider_hrnet_whmulti --arch hrnet_18 --batch_size 5 --lr 5e-4 --lr_step 15,25,35,45,55,65 --gpus 0 --num_workers 4 --dataset wider

# test
# python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume
# # flip test
# python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test 
# # multi scale test
# python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
