cd src
# train
CUDA_VISIBLE_DEVICES=1,2 python main.py cspdet --optim adam --exp_id wider_resnet18_csp_adam_change_s1_conv4_conv2 --arch cspfpn --batch_size 48 --lr 5e-4 --lr_step 70,90 --gpus 0,1 --num_workers 4 --dataset wider_csp
