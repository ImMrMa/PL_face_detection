cd src
# train
python main.py ctdet --exp_id pascal_deeplab_multi_resnet101_512 --arch deeplab_resnet101 --multi_res --dataset pascal --input_res 512 --batch_size 64 --num_epochs 70 --lr_step 45,60
# test
# python test.py ctdet --exp_id pascal_deeplab_multi_resnet101_512 --arch deeplab_resnet101 --dataset pascal --input_res 512 --batch_size 16 --resume --gpus 0,1
# # flip test
# python test.py ctdet --exp_id pascal_deeplab_multi_resnet101_512 --arch deeplab_resnet101 --dataset pascal --input_res 512 --batch_size 16 --resume --flip_test --gpus 0,1
cd ..
