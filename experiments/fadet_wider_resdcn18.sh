cd src
# train
CUDA_VISIBLE_DEVICES=2 python main.py fadet --exp_id wider_resdcn18 --arch resdcn_18 --batch_size 10 --lr 5e-4 --gpus 2 --num_workers 3 --dataset wider
# test
# python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume
# # flip test
# python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test 
# # multi scale test
# python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
