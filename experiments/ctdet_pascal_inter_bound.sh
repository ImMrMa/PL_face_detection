cd src
# train
CUDA_VISIBLE_DEVICES=1 python main.py ctdet --exp_id pascal_resnet18_inter_bound_rehm --arch resdcn_18  --dataset pascal --inter_bound --not_reg_offset --wh_weight 1  --resume --input_res 512 --batch_size 48 --num_epochs 70 --num_workers=20 --lr_step 30,45,55 --gpus 1
# test
# python test.py ctdet --exp_id pascal_resnet101_original_hm_loss --arch resdcn_101 --dataset pascal --input_res 512 --batch_size 4 --resume --gpus 1
# # # flip test
# CUDA_VISIBLE_DEVICES=1 python test.py ctdet --exp_id pascal_resnet18_inter_bound --arch resdcn_18 --dataset pascal --input_res 512 --batch_size 16 --resume --flip_test --gpus 1
cd ..
