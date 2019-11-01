cd src
# train
CUDA_VISIBLE_DEVICES=1,2 python main.py cspdet --optim adam --exp_id wider_resnet18_csp_adam_conv4_conv2_imagenet --arch cspfpn --resume --batch_size 40 --lr 2e-4 --lr_step 70,80 --gpus 0,1 --num_workers 4 --dataset wider_csp
