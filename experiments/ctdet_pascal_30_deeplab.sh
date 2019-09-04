cd src
# train
CUDA_VISIBLE_DEVICES=3 python main.py ctdet --exp_id pascal_resnet101_original_hm_loss --arch resdcn_101 --multi_res --dataset pascal --input_res 512 --batch_size 32 --num_epochs 70 --lr_step 30,45,55 --gpus 1
# test
# python test.py ctdet --exp_id pascal_deeplab_multi_resnet101_512_change_lr --arch deeplab_resnet101 --dataset pascal --input_res 512 --batch_size 16 --resume --gpus 1
# # # flip test
# python test.py ctdet --exp_id pascal_deeplab_multi_resnet101_512_change_lr --arch deeplab_resnet101 --dataset pascal --input_res 512 --batch_size 16 --resume --flip_test --gpus 1
cd ..
