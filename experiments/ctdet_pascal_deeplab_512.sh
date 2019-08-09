cd src
# train
python main.py ctdet --exp_id pascal_deeplab_resnet101_512 --arch deeplab_resnet101 --dataset pascal --input_res 512 --batch_size 20 --num_epochs 70 --lr_step 45,60
# test
python test.py ctdet --exp_id pascal_deeplab_resnet101_512 --arch deeplab_resnet101 --dataset pascal --input_res 512 --batch_size 20 --resume
# flip test
python test.py ctdet --exp_id pascal_deeplab_resnet101_512 --arch deeplab_resnet101 --dataset pascal --input_res 512 --batch_size 20 --resume --flip_test
cd ..